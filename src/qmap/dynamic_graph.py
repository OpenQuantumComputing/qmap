import time
import heapq
from enum import Enum
from functools import cmp_to_key
from dataclasses import dataclass, field
from problem_encoding import ProblemEncoding, GateEncoding
from solution import Solution, SolutionJob, sol_job_compare
from score_estimation import DepthEstimator, RemainingSwapsEstimator
from graph_automorphisms import GraphAutomorhisms


class Gate:

    def __init__(self, id : int, pair : tuple[int, int], duration : int) -> None:
        self.id = id
        self.pair = pair
        self.duration = duration

    def __repr__(self):
        return f"g{self.id}-(q{self.pair[0]},q{self.pair[1]})-{self.duration}"


class Job:

    def __init__(self, gate : Gate, hardware_pair : tuple[int, int], start : int, duration : int) -> None:
        self.gate = gate
        self.start = start
        self.hardware_pair = hardware_pair
        self.duration = duration

    def __repr__(self):
        return f"@{self.start} (h{self.hardware_pair[0]},h{self.hardware_pair[1]})-{self.duration} G={self.gate}"


class Node:

    def __init__(self, id : int, level : int, parent, job : Job, unassigned_start_gates: list[GateEncoding], qubit_hardware_map : list[int], hardware_qubit_map : list[int], hw_qubit_depth: list[int], next_qubit_gate : list[int], last_swapped_hw_qubit : list[int], nmb_rem_gates : int, least_t0_gate : int, swaps_performed: int, remaining_swaps_estimate: int, score_estimate : int, gates_layer : int) -> None:
        self.id = id
        self.level = level
        self.parent = parent
        self.job = job
        self.unassigned_start_gates = unassigned_start_gates
        self.qubit_hardware_map = qubit_hardware_map
        self.hardware_qubit_map = hardware_qubit_map
        self.hw_qubit_depth = hw_qubit_depth
        self.next_qubit_gate = next_qubit_gate
        self.last_swapped_hw_qubit = last_swapped_hw_qubit
        self.nmb_rem_gates = nmb_rem_gates
        self.least_t0_gate = least_t0_gate
        self.swaps_performed = swaps_performed
        self.remaining_swaps_estimate = remaining_swaps_estimate
        self.score_estimate = score_estimate
        self.can_use = True
        self.gates_layer = gates_layer


    def __repr__(self):
        return f'id:{self.id},parent:{"-" if self.parent == None else self.parent.id},job:{self.job},rem_jobs:{self.nmb_rem_gates},logic_to_hw:{self.qubit_hardware_map},score_est:{self.score_estimate}'


# Data class for the priority queue that ignores the second element
@dataclass(order=True)
class PrioritizedItem:
    priority: int
    # depth :int
    item: Node=field(compare=False)


class DivingParameters:
    """Defines the input parameters to setting up the diving strategy.
    There are currently two mechanisims that can trigger a dive solution.
    (1) Frequency Dive: When the number of visited nodes since the last dive is more than a given number. It does not matter which mechanism triggered the last dive.
    (2) Near Completion Dive: When the number of unplanned gates is low. If M is the currently smallest number of unplanned gates in the selected nodes, then the mechanism is
        only triggered when the number of unplanned gates in the currently selected node is equal to or less than M,
        only triggered for a specific number of nodes with M unplanned gates (the counter of such nodes is reset each time M is lowered), and
        only triggered when M is below a specific limit.
    A dive can only be triggered for a node where all logical qubits have been assigned to a hardware qubit.
    """

    diving_frequency : int
    """The least number of nodes to be visited after the last dive before a new Frequency Dive can be triggered. If negative, the Frequency Dive mechanism is disabled."""

    nmb_top_gate_dives : int
    """The maximumm number of nodes with the same number of unplanned gates that can trigger a Near Completion Dive. If not positive, Near Completion Dive mechanism is disabled."""

    max_gates_top_dive : int
    """The maximumm number of unplanned gates a node can have to trigger Near Completion Dive. This number will prevent dives too early in the algorithm. If not positive, Near Completion Dive mechanism is disabled."""

    def __init__(self, diving_frequency : int, nmb_top_gate_dives : int, max_gates_top_dive : int) -> None:
        self.diving_frequency = diving_frequency
        self.nmb_top_gate_dives = nmb_top_gate_dives
        self.max_gates_top_dive = max_gates_top_dive


class DivingStrategy:
    """Defines and handles the strategy on when to start a quick dive to a complete solution of the problem from a given node.
    There are currently two mechanisims that can trigger a dive solution.
    (1) Frequency Dive: When the number of visited nodes since the last dive is more than a given number. It does not matter which mechanism triggered the last dive.
    (2) Near Completion Dive: When the number of unplanned gates is low. If M is the currently smallest number of unplanned gates in the selected nodes, then the mechanism is
        only triggered when the number of unplanned gates in the currently selected node is equal to or less than M,
        only triggered for a specific number of nodes with M unplanned gates (the counter of such nodes is reset each time M is lowered), and
        only triggered when M is below a specific limit.
    A dive can only be triggered for a node where all logical qubits have been assigned to a hardware qubit.
    """

    parameters : DivingParameters
    """The input parameters to the diving strategy."""

    remain_selects_to_dive : int
    """The currently remaining number of nodes to be selected before a new Frequency Dive can be triggered."""

    remain_top_gate_dives : int
    """The currently remaining number of nodes that can trigger the Near Completion Dive without improving the smallest number of unplanned gates found."""

    current_top_gates : int
    """The smallest number of unplanned gates found so far for all selected nodes."""

    count_dives : int
    """Number of dives triggered so far."""

    count_frequency_dives : int
    """Number of dives triggered by the Frequency Dive mechanism so far."""

    count_gate_dives : int
    """Number of dives triggered by the Near Completion Dive mechanism so far."""

    count_loops : int
    """Total number of nodes selected so far."""

    count_improve_frequency : int
    """Number of dives triggered by the Frequency Dive mechanism that have improved the upper bound of the solution score so far."""

    count_improve_gate : int
    """Number of dives triggered by the Near Completion Dive mechanism that have improved the upper bound of the solution score so far."""

    def __init__(self, parameters : DivingParameters) -> None:
        self.parameters = parameters
        self.remain_selects_to_dive = parameters.diving_frequency
        self.remain_top_gate_dives = 0
        self.current_top_gates = 0
        self.count_dives = 0
        self.count_frequency_dives = 0
        self.count_gate_dives = 0
        self.count_loops = 0
        self.count_improve_frequency = 0
        self.count_improve_gate = 0


    def count_report(self) -> str:
        return f"Loops={self.count_loops}, Dives={self.count_dives}, By frequency={self.count_frequency_dives}, By gate={self.count_gate_dives}, Improved by freq={self.count_improve_frequency}, Improved by gate={self.count_improve_gate}"


    def can_dive(self) -> bool:
        return self._can_dive_by_frequency() or self._can_dive_by_min_gates()


    def after_dive(self, improved : bool) -> None:
        self.count_dives += 1
        self.count_loops += 1
        if self._can_dive_by_min_gates():
            self.remain_top_gate_dives -= 1
            self.count_gate_dives += 1
            if improved:
                self.count_improve_gate += 1
        if self._can_dive_by_frequency():
            self.remain_selects_to_dive = self.parameters.diving_frequency
            self.count_frequency_dives += 1
            if improved:
                self.count_improve_frequency += 1


    def after_non_dive(self) -> None:
        self.count_loops += 1
        if self.remain_selects_to_dive > 0:
            self.remain_selects_to_dive -= 1


    def min_gates_updated(self, min_gates : int) -> None:
        if min_gates <= self.parameters.max_gates_top_dive:
            self.remain_top_gate_dives = self.parameters.nmb_top_gate_dives


    def _can_dive_by_frequency(self) -> bool:
        return self.parameters.diving_frequency >= 0 and self.remain_selects_to_dive == 0


    def _can_dive_by_min_gates(self) -> bool:
        return self.remain_top_gate_dives > 0


class NodeGrouping(Enum):
    """
    How to group nodes in the node statistics
    """

    NONE = 0
    """No statistics applied"""

    LEVEL = 1
    """Nodes are grouped by their search tree level, 0 for the top node, N+1 for a child of a node on level N"""

    ADDED_SWAPS = 2
    """Nodes are grouped by the number of swap operations added"""

    REMAINING_GATES = 3
    """Nodes are grouped by the number of gates that are not planned yet"""


class NodeStatistics:

    increase_counts_gates : dict[int, dict[int, int]]
    """How much does the heuristic increase from parent node when gate is introduced. First key is number of remaining gates for parent, second key is change in heuristic, value is count."""

    increase_counts_swaps : dict[int, dict[int, int]]
    """How much does the heuristic increase from parent node when swap is introduced. First key is number of remaining gates, second key is change in heuristic, value is count."""

    heuristic_distribution : dict[int, dict[int, int]]
    """How heuristic value is distributed. First key is number of remaining gates, second key is heuristic, value is count."""

    heurisitc_exact_miss : dict[int, dict[int, int]]
    """How heuristic value misses the real score when the exact real score is known. First key is number of remaining gates, second key is real-heuristic, value is count."""

    def __init__(self, encoding : ProblemEncoding, dynamic_graph_parametes, node_grouping : NodeGrouping) -> None:

        self.encoding = encoding
        self.dynamic_graph_parametes = dynamic_graph_parametes
        self.node_grouping = node_grouping
        self.increase_counts_gates = {}
        self.increase_counts_swaps = {}
        self.heuristic_distribution = {}
        self.heurisitc_exact_miss = {}
        self.nodes_selected = 0
        self.selections_to_solution = -1
        self.extra_solutions = 0
        self.found_extra_solutions : dict[tuple[int, ...], int] = {}
        self.min_final_score : dict[int, int] = {}
        self.has_exact_score : set[bool] = set()
        self.node_children : dict[int, list[Node]] = {}
        self.excel_output = True


    def on_node_added(self, node : Node) -> None:
        self.node_children[node.id] = []
        if node.parent != None:
            self.node_children[node.parent.id].append(node)


    def on_node_selected(self, node : Node) -> None:

        self.nodes_selected += 1


    def on_terminating_node(self, node : Node) -> None:

        if self.selections_to_solution == -1:
            self.selections_to_solution = self.nodes_selected
        

    def set_final_score(self, node : Node, dyn_gr, max_score : int = -1) -> None:

        if node.id in self.has_exact_score:
            return
        should_log = False
        if should_log:
            print(f"Starts logging for special node, can_use = {node.can_use}")
        if node.nmb_rem_gates == 0:
            self.has_exact_score.add(node.id)
            self.min_final_score[node.id] = node.score_estimate
        elif not node.can_use:
            score = dyn_gr.best_score_from_node(self.dynamic_graph_parametes, node, max_score)
            self.extra_solutions += 1
            if self.extra_solutions % 1000 == 0:
                print(f"{self.extra_solutions} nodes solved extra")
            self.min_final_score[node.id] = score
            if max_score < 0 or score < max_score:
                self.has_exact_score.add(node.id)
        else:
            score = -1
            children = self.node_children[node.id]
            not_use_children = []
            for child in children:
                if child.can_use:
                    self.set_final_score(child, dyn_gr)
                    child_score = self.min_final_score[child.id]
                    if score == -1 or child_score < score:
                        score = child_score
                else:
                    not_use_children.append(child)
            for child in not_use_children:
                if score == -1 or child.score_estimate < score:
                    self.set_final_score(child, dyn_gr, score)
                    # self.set_final_score(child, dyn_gr)
                    if child.id in self.has_exact_score:
                        child_score = self.min_final_score[child.id]
                        if score == -1 or child_score < score:
                            score = child_score
            self.has_exact_score.add(node.id)
            if score == -1:
                print("Could not get score for node")
                exit(1)
            self.min_final_score[node.id] = score

        if node.can_use:
            self._add_node_statistics(node)
        if should_log:
            print(f"Ends logging for special node, can_use = {node.can_use}")


    def print_report(self) -> None:
        if self.excel_output:
            self.print_report_excel()
            return
        print(f"Nodes selected up to solution / Total number of nodes = {self.selections_to_solution}/{self.nodes_selected} = {float(self.selections_to_solution)/float(self.nodes_selected)}")
        print("Score distributions from number of remaining gates:")
        sum_all = 0
        for g in reversed(sorted(self.heuristic_distribution.keys())):
            h_d = self.heuristic_distribution[g]
            sum_g = sum(h_d.values())
            sum_all += sum_g
            print(f"{g}\t{sum_g}\t" + ", ".join(f"{sc}:{h_d[sc]}" for sc in sorted(h_d.keys())))
        print(f"Contributirs to score distributions = {sum_all}")
        print("Score increase distributions in children from number of remaining gates (split by nodes adding Gate/Swap):")
        sum_all_g = 0
        sum_all_s = 0
        for g in reversed(sorted(set(self.increase_counts_gates.keys()).union(self.increase_counts_swaps.keys()))):
            inc_c_g = self.increase_counts_gates[g] if g in self.increase_counts_gates else {}
            inc_c_s = self.increase_counts_swaps[g] if g in self.increase_counts_swaps else {}
            sum_g = sum(inc_c_g.values())
            sum_s = sum(inc_c_s.values())
            sum_all_g += sum_g
            sum_all_s += sum_s
            scores = set(inc_c_g.keys()).union(inc_c_s.keys())
            print(f"{g}\t{('-' if sum_g == 0 else sum_g)}/{('-' if sum_s == 0 else sum_s)}\t" + ", ".join(f"{sc}:{(inc_c_g[sc] if sc in inc_c_g else '-')}/{(inc_c_s[sc] if sc in inc_c_s else '-')}" for sc in sorted(scores)))
        print(f"Contributirs to score increas distributions = {sum_all_g}+{sum_all_s}={sum_all_g + sum_all_s}")
        print("Estimated deviations for the heuristic scores, exact score known:")
        sum_all = 0
        for g in reversed(sorted(self.heurisitc_exact_miss.keys())):
            h_e_m = self.heurisitc_exact_miss[g]
            sum_g = sum(h_e_m.values())
            sum_all += sum_g
            print(f"{g}\t{sum(h_e_m.values())}\t" + ", ".join(f"{sc}:{h_e_m[sc]}" for sc in sorted(h_e_m.keys())))
        print(f"Contributirs to estimated deviations for the heuristic scores = {sum_all}")


    def print_report_excel(self) -> None:
        self._print_table(self.heuristic_distribution, "Heuristic score distributions", "Score")
        self._print_table(self.increase_counts_gates, "Distribution of increase in heuristic score in children, gates only", "Increase")
        self._print_table(self.increase_counts_swaps, "Distribution of increase in heuristic score in children, swaps only", "Increase")
        increase_counts : dict[int, dict[int, int]] = {}
        for key1 in set(self.increase_counts_gates.keys()).union(self.increase_counts_swaps.keys()):
            inc_c_gates = self.increase_counts_gates[key1] if key1 in self.increase_counts_gates else {}
            inc_c_swaps = self.increase_counts_swaps[key1] if key1 in self.increase_counts_swaps else {}
            summed_counts : dict[int, int] = {}
            for key2 in set(inc_c_gates.keys()).union(inc_c_swaps.keys()):
                inc_c_sum = (inc_c_gates[key2] if key2 in inc_c_gates else 0) + (inc_c_swaps[key2] if key2 in inc_c_swaps else 0)
                summed_counts[key2] = inc_c_sum
            increase_counts[key1] = summed_counts
        self._print_table(increase_counts, "Distribution of increase in heuristic score in children, gates and swaps", "Increase")
        self._print_table(self.heurisitc_exact_miss, "Distribution of heuristic score deviation from exact score", "Deviation")


    def _print_table(self, table : dict[int, dict[int, int]], heading : str, columns_heading):
        print()
        print(heading)
        print(f"\t\t{columns_heading}")
        min_col = min(min(v.keys()) for v in table.values())
        max_col = max(max(v.keys()) for v in table.values())
        col_values = list(range(min_col, max_col + 1))
        sum_all = 0
        print(f"{self._grouping_row_heading()}\tSum\t" + "\t".join(str(col_v) for col_v in col_values))
        for key in self._statistics_rows(table):
            row_values = table[key]
            sum_row = sum(row_values.values())
            sum_all += sum_row
            print(f"{key}\t{sum_row}\t" + "\t".join((str(row_values[c]) if c in row_values else "") for c in col_values))
        print(f"Sum\t{sum_all}")


    def _grouping(self, node : Node) -> int:

        match self.node_grouping:
            case NodeGrouping.LEVEL:
                return node.level
            case NodeGrouping.ADDED_SWAPS:
                return node.swaps_performed
            case NodeGrouping.REMAINING_GATES:
                return node.nmb_rem_gates
            case _:
                raise ValueError(f"Unknown node grouping : {self.node_grouping}")


    def _grouping_row_heading(self) -> str:
        match self.node_grouping:
            case NodeGrouping.LEVEL:
                return "Level"
            case NodeGrouping.ADDED_SWAPS:
                return "Added swaps"
            case NodeGrouping.REMAINING_GATES:
                return "Remaining gates"


    def _statistics_rows(self, table : dict[int, dict[int, int]]) -> list[int]:
        if self.node_grouping == NodeGrouping.REMAINING_GATES:
            return list(reversed(sorted(table.keys())))
        else:
            return list(sorted(table.keys()))


    def _add_node_statistics(self, node : Node) -> None:
        self._add_heuristic_distribution(node)
        if node.parent != None:
            self._add_increase_distribution(node.parent, node)
        self._add_heuristic_deviation(node)


    def _add_heuristic_distribution(self, node : Node) -> None:

        group_idx = self._grouping(node)
        score_node = node.score_estimate
        if group_idx in self.heuristic_distribution:
            h_d = self.heuristic_distribution[group_idx]
        else:
            h_d = {}
            self.heuristic_distribution[group_idx] = h_d
        h_d[score_node] = 1 if not score_node in h_d else h_d[score_node] + 1


    def _add_increase_distribution(self, parent : Node, child : Node) -> None:

        group_idx = self._grouping(parent)
        diff_score = child.score_estimate - parent.score_estimate
        increase_counts = self.increase_counts_swaps if parent.nmb_rem_gates == child.nmb_rem_gates else self.increase_counts_gates
        if group_idx in increase_counts:
            incr_c = increase_counts[group_idx]
        else:
            incr_c = {}
            increase_counts[group_idx] = incr_c
        incr_c[diff_score] = 1 if not diff_score in incr_c else incr_c[diff_score] + 1


    def _add_heuristic_deviation(self, node : Node) -> None:

        group_idx = self._grouping(node)
        score_incr = self.min_final_score[node.id] - node.score_estimate
        if group_idx in self.heurisitc_exact_miss:
            h_e_m = self.heurisitc_exact_miss[group_idx]
        else:
            h_e_m = {}
            self.heurisitc_exact_miss[group_idx] = h_e_m
        h_e_m[score_incr] = 1 if not score_incr in h_e_m else h_e_m[score_incr] + 1


    def _print_node(self, node : Node) -> str:
        rem_gates = []
        for q in range(len(self.encoding.nmb_qubit_gates)):
            for idx in range(node.next_qubit_gate[q], self.encoding.nmb_qubit_gates[q]):
                gate = self.encoding.qubit_gates[q][idx]
                if gate.pair[0] == q:
                    rem_gates.append(gate)
        rem_gates_str = "|".join(str(g) for g in rem_gates) if rem_gates else "-"
        return f"{node},hw_depth:{node.hw_qubit_depth},min_final_score:{self.min_final_score[node.id] if node.id in self.min_final_score else '?'},has_exact_score:{node.id in self.has_exact_score},can_use:{node.can_use},remains:{rem_gates_str}"


class DynamicGraphParameters:

    def __init__(self, diving_parameters : DivingParameters, swaps_objective : int, depth_objective : int, unassigned_objective : int, beam_width : int, remain_gates_weight : int, statistics_grouping : NodeGrouping, layering_active : bool, timeout_seconds : float) -> None:

        self.diving_parameters = diving_parameters
        self.swaps_objective = swaps_objective
        self.depth_objective = depth_objective
        self.unassigned_objective = unassigned_objective
        self.beam_width = beam_width
        self.remain_gates_weight = remain_gates_weight
        self.statistics_grouping = statistics_grouping
        self.layering_active = layering_active
        self.timeout_seconds = timeout_seconds


class TerminationCause(Enum):
    """
    Reason for why the dynamic graph solver algoriothm was terminated.
    """

    SOLUTIONFOUND = 0
    """The algorithm found a solution (not including solutions from the diving mechanism)."""

    EXHAUSTED = 1
    """All nodes were visited, no solution was found. Should not happen to a solvable circuit problem."""

    TIMEOUT = 2
    """The timout limit was reached before the algorithm could find a final solution."""


class DynamicGraph:

    def __init__(self, encoding : ProblemEncoding, parameters : DynamicGraphParameters, depth_estimator : DepthEstimator, swap_estimator : RemainingSwapsEstimator) -> None:

        self.encoding = encoding
        self.problem = encoding.problem
        self.depth_estimator = depth_estimator
        self.swap_estimator = swap_estimator
        self.diving_strategy = DivingStrategy(parameters.diving_parameters)
        self.swaps_objective = parameters.swaps_objective
        self.depth_objective = parameters.depth_objective
        self.unassigned_objective = parameters.unassigned_objective
        self.remain_gates_weight = parameters.remain_gates_weight
        self.beam_width = parameters.beam_width
        self.layering_active = parameters.layering_active
        if self.layering_active:
            self.gate_layer = self.encoding.build_gate_layers()
            self.qubit_gate_layer = self._build_qubit_gate_layers()
        self.timeout_seconds = parameters.timeout_seconds

        # Initialize queue
        self.queue = []
        self.added_nodes_by_signatures : dict[tuple[int, ...], list[tuple[list[int], Node]]] = {}
        self.nodes_selected = 0
        self.nodes_added = 0
        self.next_node_id = 0
        self.origin : Node = None
        self.current_best_solution : tuple[int, Solution] = None
        self.statistics : NodeStatistics = None if parameters.statistics_grouping == NodeGrouping.NONE else NodeStatistics(self.encoding, self._node_statistics_dynamic_graph_parameters(), parameters.statistics_grouping)

        graph_aut = GraphAutomorhisms(self.encoding.hardware_edges)
        graph_aut.get_edge_symmetries()
        self.first_gate_edges = graph_aut.non_isomorphic_edges


    def _node_statistics_dynamic_graph_parameters(self) -> DynamicGraphParameters:

        return DynamicGraphParameters(DivingParameters(-1, 0, 0), self.swaps_objective, self.depth_objective, self.unassigned_objective, 0, self.remain_gates_weight, NodeGrouping.NONE, self.layering_active, 0.0)


    def _build_qubit_gate_layers(self) -> list[list[int]]:

        return [[self.gate_layer[gate.id] for gate in gates] for gates in self.encoding.qubit_gates]


    def _build_qubit_hardware_map_(self, hardware_to_qubit_names : dict[str, str]) -> list[int]:

        qubit_hardware_map = [-1] * self.encoding.nmb_logical_qubits
        if hardware_to_qubit_names != None:
            hw_name_to_q = {}
            lo_name_to_q = {}
            for idx in range(self.encoding.nmb_hardware_qubits):
                hw_name_to_q[self.encoding.hardware_qubit_names[idx]] = idx
            for idx in range(self.encoding.nmb_logical_qubits):
                lo_name_to_q[self.encoding.logical_qubit_names[idx]] = idx
            for hw_name, lo_name in hardware_to_qubit_names.items():
                qubit_hardware_map[lo_name_to_q[lo_name]] = hw_name_to_q[hw_name]
        return qubit_hardware_map


    def _add_gate_node(self, parent : Node, gate : GateEncoding, hw_from : int, hw_to : int, place_in_queue : bool, assign_to_opposite : bool = False) -> Node:

        if self.layering_active and self.qubit_gate_layer[gate.pair[0]][self.encoding.gate_position_for_qubit[gate.id][0]] > parent.gates_layer:
            return None

        q_from = parent.hardware_qubit_map[hw_from]
        q_to = parent.hardware_qubit_map[hw_to]
        max_qubit_depth = max(parent.hw_qubit_depth[hw_from], parent.hw_qubit_depth[hw_to])

        if q_from == -1 and q_to == -1:
            unassigned_start_gates = list(j for j in parent.unassigned_start_gates if j.id != gate.id)
        else:
            unassigned_start_gates = parent.unassigned_start_gates

        if q_from == -1 and q_to == -1:
            new_q_from = gate.pair[1] if assign_to_opposite else gate.pair[0]
            new_q_to = gate.pair[0] if assign_to_opposite else gate.pair[1]
            qubit_hardware_map = parent.qubit_hardware_map.copy()
            qubit_hardware_map[new_q_from] = hw_from
            qubit_hardware_map[new_q_to] = hw_to
            pass
        elif q_from == -1:
            new_q_from = gate.pair[0] if q_to == gate.pair[1] else gate.pair[1]
            new_q_to = q_to
            qubit_hardware_map = parent.qubit_hardware_map.copy()
            qubit_hardware_map[new_q_from] = hw_from
        elif q_to == -1:
            new_q_to = gate.pair[0] if q_from == gate.pair[1] else gate.pair[1]
            new_q_from = q_from
            qubit_hardware_map = parent.qubit_hardware_map.copy()
            qubit_hardware_map[new_q_to] = hw_to
        else:
            new_q_from = q_from
            new_q_to = q_to
            qubit_hardware_map = parent.qubit_hardware_map

        hardware_qubit_map = parent.hardware_qubit_map if q_from != -1 and q_to != -1 else None
        next_qubit_gate = parent.next_qubit_gate.copy()
        next_qubit_gate[new_q_from] += 1
        next_qubit_gate[new_q_to] += 1
        hw_qubit_depth = parent.hw_qubit_depth.copy()
        hw_qubit_depth[hw_from] = hw_qubit_depth[hw_to] = max_qubit_depth + gate.duration
        last_swapped_hw_qubit = parent.last_swapped_hw_qubit.copy()
        last_swapped_hw_qubit[hw_from] = last_swapped_hw_qubit[hw_to] = -1
        remain_swaps_est = 0 if self.swaps_objective == 0 else self.swap_estimator.remain_swaps_estimate(next_qubit_gate, qubit_hardware_map)
        job = Job(gate, (hw_from, hw_to), max_qubit_depth, gate.duration)
        least_t0_gate = gate.id if parent.level == 0 else parent.least_t0_gate

        if not self.layering_active:
            gates_layer = 0
        elif parent.nmb_rem_gates == 1 or all(self.qubit_gate_layer[idx][next_qubit_gate[idx]] > parent.gates_layer for idx in range(self.encoding.nmb_logical_qubits) if next_qubit_gate[idx] < self.encoding.nmb_qubit_gates[idx]):
            gates_layer = parent.gates_layer + 1
        else:
            gates_layer = parent.gates_layer
        return self._add_node(parent, job, unassigned_start_gates, qubit_hardware_map, hardware_qubit_map, hw_qubit_depth, next_qubit_gate, last_swapped_hw_qubit, parent.nmb_rem_gates - 1, least_t0_gate, parent.swaps_performed, remain_swaps_est, gates_layer, place_in_queue)


    def _add_swap_node(self, parent : Node, hw_from : int, hw_to : int, place_in_queue : bool) -> Node:

        q_from = parent.hardware_qubit_map[hw_from]
        q_to = parent.hardware_qubit_map[hw_to]
        max_qubit_depth = max(parent.hw_qubit_depth[hw_from], parent.hw_qubit_depth[hw_to])
        swap_job = Job(None, (hw_from, hw_to), max_qubit_depth, self.problem.swap_time)
        qubit_hardware_map = parent.qubit_hardware_map.copy()
        if q_from != -1:
            qubit_hardware_map[q_from] = hw_to
        if q_to != -1:
            qubit_hardware_map[q_to] = hw_from
        hw_qubit_depth = parent.hw_qubit_depth.copy()
        hw_qubit_depth[hw_from] = hw_qubit_depth[hw_to] = max_qubit_depth + self.problem.swap_time
        last_swapped_hw_qubit = parent.last_swapped_hw_qubit.copy()
        last_swapped_hw_qubit[hw_from] = hw_to
        last_swapped_hw_qubit[hw_to] = hw_from
        remain_swaps_est = 0 if self.swaps_objective == 0 else self.swap_estimator.remain_swaps_estimate(parent.next_qubit_gate, qubit_hardware_map)
        return self._add_node(parent, swap_job, parent.unassigned_start_gates, qubit_hardware_map, None, hw_qubit_depth, parent.next_qubit_gate, last_swapped_hw_qubit, parent.nmb_rem_gates, parent.least_t0_gate, parent.swaps_performed + 1, remain_swaps_est, parent.gates_layer, place_in_queue)


    def _remove_from_pareto(self, node : Node) -> None:

        signature = tuple(node.qubit_hardware_map + node.next_qubit_gate)
        pareto = self.added_nodes_by_signatures[signature]
        n_before = len(pareto)
        new_pareto = [ p for p in pareto if p[1] != node]
        n_after = len(new_pareto)
        if n_before != n_after + 1:
            raise Exception(f"Pareto length should decrease by one, but changed from {n_before} to {n_after}")
        if new_pareto:
            self.added_nodes_by_signatures[signature] = new_pareto
        else:
            del self.added_nodes_by_signatures[signature]


    def _add_node(self, parent : Node, job : Job, unassigned_start_gates: list[GateEncoding], qubit_hardware_map : list[int], hardware_qubit_map : list[int], hw_qubit_depth: list[int], next_qubit_gate : list[int], last_swapped_hw_qubit : list[int], nmb_rem_gates : int, least_t0_gate : int, swaps_performed: int, remaining_swaps_estimate: int, gates_layer : int, place_in_queue : bool) -> Node:

        if place_in_queue:
            signature = tuple(qubit_hardware_map + next_qubit_gate)
            pareto_array = ([] if self.depth_objective == 0 else hw_qubit_depth) + ([] if self.swaps_objective == 0 else [swaps_performed])
            if signature in self.added_nodes_by_signatures:
                pareto = self.added_nodes_by_signatures[signature]
                for idx in range(len(pareto)):
                    cmp_vector = pareto[idx][0]
                    cmp = 0  # -2 or -1 if cmp_vector lexiographically before, +1, +2 if opposite, 0 if equal. -2 and +2 are for stric domination.
                    for q in range(len(pareto_array)):
                        if cmp_vector[q] < pareto_array[q]:
                            if cmp == 2:
                                cmp = 1
                                break
                            else:
                                cmp = -2
                        elif cmp_vector[q] > pareto_array[q]:
                            if cmp == -2:
                                cmp = -1
                                break
                            else:
                                cmp = 2

                    if cmp == 0:
                        # New node is identical to the one in pareto front, no need to add
                        if self.statistics != None and parent != None:
                            self.statistics.node_children[parent.id].append(pareto[idx][1])
                        return None
                    elif cmp == -2:
                        # New node is totally dominated
                        if self.statistics != None and parent != None:
                            level = 0 if parent == None else parent.level + 1
                            if hardware_qubit_map == None:
                                hardware_qubit_map = [-1] * self.encoding.nmb_hardware_qubits
                                for qubit, hw_qubit in enumerate(qubit_hardware_map):
                                    if hw_qubit != -1:
                                        hardware_qubit_map[hw_qubit] = qubit
                            depth_score_est = 0 if self.depth_objective == 0 else self.depth_objective * self.depth_estimator.depth_estimate(hw_qubit_depth, next_qubit_gate, qubit_hardware_map, hardware_qubit_map)
                            swaps_score_est = self.swaps_objective * (swaps_performed + remaining_swaps_estimate)
                            unassigned_qubits_score = 0 if self.unassigned_objective == 0 else self.unassigned_objective * sum(1 for hw in qubit_hardware_map if hw == -1)
                            node = Node(self.next_node_id, level, parent, job, unassigned_start_gates, qubit_hardware_map, hardware_qubit_map, hw_qubit_depth, next_qubit_gate, last_swapped_hw_qubit, nmb_rem_gates, least_t0_gate, swaps_performed, remaining_swaps_estimate, depth_score_est + swaps_score_est + unassigned_qubits_score, gates_layer)
                            node.can_use = False
                            self.next_node_id += 1
                            self.statistics.on_node_added(node)
                        return None
                    elif cmp != -1:
                        # New node is part of pareto front and should be inserted at position idx
                        break
                    # else: New node comes lexiographically after all nodes in pareto front tested so far, but is not dominated by any of them

            else:
                pareto = []
                self.added_nodes_by_signatures[signature] = pareto
                cmp = -1
                idx = 0

        # New node should be created.
        if hardware_qubit_map == None:
            hardware_qubit_map = [-1] * self.encoding.nmb_hardware_qubits
            for qubit, hw_qubit in enumerate(qubit_hardware_map):
                if hw_qubit != -1:
                    hardware_qubit_map[hw_qubit] = qubit
        level = 0 if parent == None else parent.level + 1
        depth_score_est = 0 if self.depth_objective == 0 else self.depth_objective * self.depth_estimator.depth_estimate(hw_qubit_depth, next_qubit_gate, qubit_hardware_map, hardware_qubit_map)
        swaps_score_est = self.swaps_objective * (swaps_performed + remaining_swaps_estimate)
        unassigned_qubits_score = 0 if self.unassigned_objective == 0 else self.unassigned_objective * sum(1 for hw in qubit_hardware_map if hw == -1)

        node = Node(self.next_node_id, level, parent, job, unassigned_start_gates, qubit_hardware_map, hardware_qubit_map, hw_qubit_depth, next_qubit_gate, last_swapped_hw_qubit, nmb_rem_gates, least_t0_gate, swaps_performed, remaining_swaps_estimate, depth_score_est + swaps_score_est + unassigned_qubits_score, gates_layer)
        self.next_node_id += 1
        heap_key = node.score_estimate + self.remain_gates_weight * node.nmb_rem_gates
        heapq.heappush(self.queue, PrioritizedItem(heap_key, node))
        self.nodes_added += 1
        if self.statistics != None and place_in_queue:
            self.statistics.on_node_added(node)

        if place_in_queue:
            # New node placed in pareto front. cmp is either -1 (add to end of pareto front), +1 (insert at idx without removing next) or +2 (replace node at idx)
            if cmp == -1:
                pareto.append((pareto_array, node))
            else:
                if cmp == 1:
                    pareto.insert(idx, (pareto_array, node))
                    idx += 2
                else:
                    pareto[idx][1].can_use = False
                    pareto[idx] = (pareto_array, node)
                    idx += 1
                for idx2 in reversed(range(idx, len(pareto))):
                    cmp_vector = pareto[idx2][0]
                    if all(pareto_array[q] <= cmp_vector[q] for q in range(len(pareto_array))):
                        pareto[idx2][1].can_use = False
                        pareto.pop(idx2)
                
        # print(f"Added node: {node}")

        return node


    def _select_node(self) -> Node:

        selected_node : Node = heapq.heappop(self.queue).item
        self.nodes_selected += 1
        # print(f"Selected node level {selected_node.level}: {selected_node}")

        return selected_node


    def _expand_node(self, node_to_expand : Node):

        # Expand the node with possible jobs and swaps
        swaps = []
        edges = self.first_gate_edges if node_to_expand.level == 0 else self.encoding.hardware_edges
        for hw_from, hw_to in edges:
            if node_to_expand.last_swapped_hw_qubit[hw_from] != hw_to or node_to_expand.last_swapped_hw_qubit[hw_to] != hw_from:
                at_t0 = node_to_expand.hw_qubit_depth[hw_from] == 0 and node_to_expand.hw_qubit_depth[hw_to] == 0
                q_from = node_to_expand.hardware_qubit_map[hw_from]
                q_to = node_to_expand.hardware_qubit_map[hw_to]

                if q_from == -1 and q_to == -1:
                    # No hardware qubits are assigned to logical qubits
                    for gate in node_to_expand.unassigned_start_gates:
                        if not at_t0 or gate.id >= node_to_expand.least_t0_gate:
                            self._add_gate_node(node_to_expand, gate, hw_from, hw_to, True, False)
                            q0 = gate.pair[0]
                            q1 = gate.pair[1]
                            if node_to_expand.level > 0 and(self.encoding.nmb_qubit_gates[q0] != 1 or self.encoding.nmb_qubit_gates[q1] != 1):
                                self._add_gate_node(node_to_expand, gate, hw_from, hw_to, True, True)

                elif q_from == -1 or q_to == -1:
                    # One hardware qubit is assigned to a logical qubit, the other is not
                    q_as = q_from if q_to == -1 else q_to
                    next_q_as_gate = node_to_expand.next_qubit_gate[q_as]
                    if next_q_as_gate < self.encoding.nmb_qubit_gates[q_as]:
                        gate = self.encoding.qubit_gates[q_as][next_q_as_gate]
                        q_unas = gate.pair[1] if gate.pair[0] == q_as else gate.pair[0]
                        if self.encoding.qubit_gates[q_unas][0] == gate and(not at_t0 or gate.id >= node_to_expand.least_t0_gate):
                            self._add_gate_node(node_to_expand, gate, hw_from, hw_to, True)
                    swaps.append((hw_from, hw_to))

                else:
                    # Both hardware qubits are assigned to logical qubits
                    next_gate_from = node_to_expand.next_qubit_gate[q_from]
                    next_job_to = node_to_expand.next_qubit_gate[q_to]
                    if next_gate_from < self.encoding.nmb_qubit_gates[q_from]:
                        gate = self.encoding.qubit_gates[q_from][next_gate_from]
                        if q_to in gate.pair and self.encoding.qubit_gates[q_to][next_job_to] == gate and(not at_t0 or gate.id >= node_to_expand.least_t0_gate):
                            # Qubits on edge have the same next job, add as node
                            self._add_gate_node(node_to_expand, gate, hw_from, hw_to, True)
                    if next_gate_from < self.encoding.nmb_qubit_gates[q_from] or next_job_to < self.encoding.nmb_qubit_gates[q_to]:
                        # At least one of the qubits have remaining jobs, might be useful to swap
                        swaps.append((hw_from, hw_to))

        # Add possible swaps
        for hw_from, hw_to in swaps:
            self._add_swap_node(node_to_expand, hw_from, hw_to, True)


    def _first_gates(self, node: Node) -> list[GateEncoding]:

        gates = []
        for q in range(self.encoding.nmb_logical_qubits):
            if node.next_qubit_gate[q] < self.encoding.nmb_qubit_gates[q]:
                gate = self.encoding.qubit_gates[q][node.next_qubit_gate[q]]
                if q == gate.pair[0]:
                    q_other = gate.pair[1]
                    if node.next_qubit_gate[q_other] == self.encoding.gate_position_for_qubit[gate.id][1]:
                        gates.append(gate)
        return gates


    def _dive_to_solution(self, node: Node) -> tuple[int, Solution]:

        # The diving algorithm searches for a quick path to a solution from the input node, without ensuring optimality.
        # The algorithm will make swaps that improve the total distance between the quibits in the gates that are ready to be planned,
        # until some of these gates have neighbour qubits on the graph, then add the gates and continue until all remaining gates are planned.

        while node.nmb_rem_gates > 0:

            # Get gates that are the next remaining job for both of their qubits. Reduce set of gates to those with lowest layer if layering is active.
            first_gates = self._first_gates(node)
            if self.layering_active:
                min_layer = min(self.gate_layer[g.id] for g in first_gates)
                first_gates = [g for g in first_gates if self.gate_layer[g.id] == min_layer]
            ready_gates : list[GateEncoding] = []

            # Use swaps to improve distance between qubits in first_gates until some of them are neighbours on hardware.
            while not ready_gates:

                # Calculate qubit distance and see if there are gates ready to be planned.
                sum_dist = 0
                min_single_dist = -1
                ready_gates = []
                for g in first_gates:
                    dist = self.swap_estimator.hw_distances.swap_distance[node.qubit_hardware_map[g.pair[0]]][node.qubit_hardware_map[g.pair[1]]]
                    if dist == 0:
                        ready_gates.append(g)
                    else:
                        sum_dist += dist
                        if min_single_dist == -1 or dist < min_single_dist:
                            min_single_dist = dist

                if not ready_gates:
                    # No gates can be planned now. Search for most improving swap.
                    best_sum_dist = sum_dist
                    best_min_single_dist = min_single_dist
                    best_swap : tuple[int, int] = None

                    for hw_from, hw_to in self.encoding.hardware_edges:
                        q_from = node.hardware_qubit_map[hw_from]
                        q_to = node.hardware_qubit_map[hw_to]
                        if q_from != -1 or q_to != -1:
                            qubit_hardware_map = node.qubit_hardware_map.copy()
                            if q_from != -1:
                                qubit_hardware_map[q_from] = hw_to
                            if q_to != -1:
                                qubit_hardware_map[q_to] = hw_from
                            distances = list(self.swap_estimator.hw_distances.swap_distance[qubit_hardware_map[g.pair[0]]][qubit_hardware_map[g.pair[1]]] for g in first_gates)
                            swap_dist = sum(distances)
                            swap_min_single_dist = min(distances)
                            if swap_dist < best_sum_dist or (swap_dist == best_sum_dist and swap_min_single_dist < best_min_single_dist):
                                best_sum_dist = swap_dist
                                best_min_single_dist = swap_min_single_dist
                                best_swap = (hw_from, hw_to)

                    if best_swap == None:
                        print(first_gates)
                        print(sum_dist)
                        raise Exception("Not possible to bring qubits together")
                    else:
                        # Best improving swap is found, apply and retry
                        node = self._add_swap_node(node, best_swap[0], best_swap[1], False)

                else:
                    # Some gates can be planned now
                    for gate in ready_gates:
                        h1 = node.qubit_hardware_map[gate.pair[0]]
                        h2 = node.qubit_hardware_map[gate.pair[1]]
                        hw_from, hw_to = next((hw1, hw2) for hw1, hw2 in self.encoding.hardware_edges if (h1 == hw1 and h2 == hw2) or (h1 == hw2 and h2 == hw1))
                        node = self._add_gate_node(node, gate, hw_from, hw_to, False)

        return (node.score_estimate, self._create_solution(node, False))


    def _create_solution(self, end_node: Node, optimality_ensured : bool) -> Solution:

        hardware_to_qubit = end_node.hardware_qubit_map.copy()
        n = end_node
        sol_jobs : list[SolutionJob] = []
        while n != None:
            job = n.job
            if job != None:
                if job.gate == None:
                    hw_from = job.hardware_pair[0]
                    hw_to = job.hardware_pair[1]
                    q_from = hardware_to_qubit[hw_to]
                    q_to = hardware_to_qubit[hw_from]
                    hardware_to_qubit[hw_from] = q_from
                    hardware_to_qubit[hw_to] = q_to
                    prob_gate = None
                else:
                    hw_from_idx = 1 if hardware_to_qubit[job.hardware_pair[0]] == job.gate.pair[1] else 0
                    hw_from = job.hardware_pair[hw_from_idx]
                    hw_to = job.hardware_pair[1 - hw_from_idx]
                    prob_gate = self.problem.gates[job.gate.id]
                sol_jobs.append(SolutionJob(job.start, job.duration, prob_gate, (self.encoding.hardware_qubit_names[hw_from], self.encoding.hardware_qubit_names[hw_to])))
            n = n.parent
        
        sol_jobs.sort(key = cmp_to_key(sol_job_compare))

        # In case there are logical qubits that are not assigned to any hardware qubit because they do no have any gates, assign then to unused hardware qubits
        unused_hw = [hw_q for hw_q, lo_q in enumerate(hardware_to_qubit) if lo_q < 0]
        unused_lo = [q for q in range(self.encoding.nmb_logical_qubits) if not q in hardware_to_qubit]
        for idx in range(min(len(unused_hw), len(unused_lo))):
            hardware_to_qubit[unused_hw[idx]] = unused_lo[idx]

        hardware_to_qubit_names = {self.encoding.hardware_qubit_names[hw_q]: self.encoding.logical_qubit_names[lo_q] for hw_q, lo_q in enumerate(hardware_to_qubit) if lo_q >= 0}
        return Solution(self.problem, optimality_ensured, hardware_to_qubit_names, sol_jobs)


    def best_score_from_node(self, parameters : DynamicGraphParameters, node : Node, max_score : int) -> int:

        dyn_gr = DynamicGraph(self.encoding, parameters, self.depth_estimator, self.swap_estimator)
        self.origin = dyn_gr._add_node(None, None, node.unassigned_start_gates, node.qubit_hardware_map, node.hardware_qubit_map, node.hw_qubit_depth, node.next_qubit_gate, node.last_swapped_hw_qubit, node.nmb_rem_gates, node.least_t0_gate, node.swaps_performed, node.remaining_swaps_estimate, node.gates_layer, True)
        solution_node, _ = dyn_gr._solve(False, max_score)
        return max_score if solution_node == None else solution_node.score_estimate


    def find_solution(self, pre_assigned : dict[str, str]) -> tuple[Solution, TerminationCause]:

        # Add origin to queue
        all_qubits = list(range(self.encoding.nmb_logical_qubits))
        qubit_to_hardware = self._build_qubit_hardware_map_(pre_assigned)
        first_gates = [gate for gate in self.encoding.gates if self.encoding.gate_position_for_qubit[gate.id] == (0, 0)]
        hw_qubit_depth = [0] * self.encoding.nmb_hardware_qubits
        next_qubit_gate = [0] * self.encoding.nmb_logical_qubits
        last_swapped_hw_qubit = [-1] * self.encoding.nmb_hardware_qubits
        remain_swaps_est = 0 if self.swaps_objective == 0 else self.swap_estimator.remain_swaps_estimate(next_qubit_gate, qubit_to_hardware)
        self.origin = self._add_node(None, None, first_gates, qubit_to_hardware, None, hw_qubit_depth, next_qubit_gate, last_swapped_hw_qubit, len(self.encoding.gates), 0, 0, remain_swaps_est, 0, True)

        # Find solution
        solution_node, cause = self._solve(True)
        if solution_node != None:
            solution = self._create_solution(solution_node, True)
        elif self.current_best_solution != None:
            solution = self.current_best_solution[1]
        else:
            solution = None
        return (solution, cause)


    def _solve(self, log : bool, max_score : int = -1) -> tuple[Node, TerminationCause]:

        t_step = 5.0
        t_start = time.time()
        t_next_log = t_start + t_step
        t_ellapsed = 0.0
        max_level = 0
        total_gates = len(self.encoding.gates)
        min_remaining_gates = total_gates
        has_logged = False
        running_statistics = self.statistics != None
        is_beam = self.beam_width > 0
        if log:
            if self.remain_gates_weight == 0:
                adm_log_msg = ", admissible heuristic"
            else:
                adm_log_msg = ", non-admissible heuristic"
            if is_beam:
                beam_log_msg = f", beam width {self.beam_width}"
            else:
                beam_log_msg = ""
            print(f"Starting tree search{adm_log_msg}{beam_log_msg}")

        beam_level = 0
        if log and is_beam:
            print(f"Totally {total_gates} gates to be placed")
            print("    Sec   Lev  Nodes  Selected Rem-g Up-B")

        do_next_level = True
        while do_next_level:

            if is_beam:
                if log and beam_level % 10 == 0:
                    t_ellapsed = time.time() - t_start
                    depth_ub = "   NA" if self.current_best_solution == None else "{:5d}".format(self.current_best_solution[0])
                    print("{:7.1f}".format(t_ellapsed) + "{:6d}".format(beam_level) + "{:7d}".format(len(self.queue)) + "{:10d}".format(self.nodes_selected) + "{:6d}".format(min_remaining_gates) + depth_ub)

                # Only keep nodes inside beam
                source_queue : list[Node] = []
                moved = 0
                while self.queue and moved < self.beam_width:
                    source_queue.append(heapq.heappop(self.queue).item)
                    moved = moved + 1

                # Nodes outside beam are rejected and from the pareto front of their signatures
                for queue_item in self.queue:
                    selected_node : Node = queue_item.item
                    if selected_node.can_use:
                        self._remove_from_pareto(selected_node)

                self.queue = []

            else:
                source_queue = self.queue
                do_next_level = False

            beam_level = beam_level + 1

            while source_queue:

                # Select
                t_now = time.time()
                t_ellapsed = t_now - t_start
                if self.timeout_seconds > 0.0 and t_ellapsed > self.timeout_seconds:
                    if log:
                        if self.current_best_solution == None:
                            print(f"Algorithm timed out, no solution found")
                            print(f"Weight unassigned qubits = {self.unassigned_objective}")
                        else:
                            best_sol = self.current_best_solution[1]
                            print(f"Algorithm timed out, using best solution found up to timeout")
                            print(f"Weight unassigned qubits = {self.unassigned_objective}")
                            print(f"Swaps = {best_sol.swaps()}, Weight = {self.swaps_objective}")
                            print(f"Depth = {best_sol.depth()}, Weight = {self.depth_objective}")
                            print(f"Score = {self.current_best_solution[0]}")
                        print(f"Nodes added = {self.nodes_added}")
                        print(f"Nodes selected = {self.nodes_selected}")
                        print(f"Execution time = {t_ellapsed} sec")
                    return (None, TerminationCause.TIMEOUT)
                        
                if is_beam:
                    selected_node = source_queue.pop(0)
                    self.nodes_selected += 1
                else:
                    selected_node = self._select_node()

                if max_score >= 0 and selected_node.score_estimate >= max_score:
                    do_next_level = False
                    break

                if not selected_node.can_use:
                    # This node is to be regarded as removed from the queue, a better state of the same signature has been found.
                    continue

                if running_statistics:
                    self.statistics.on_node_selected(selected_node)

                # Terminate with assured best solution if all gates placed and we are not doing statistics
                if selected_node.nmb_rem_gates == 0:
                    if running_statistics:
                        self.statistics.on_terminating_node(selected_node)
                    else:
                        if log:
                            print(f"Solution found")
                            print(f"Weight unassigned qubits = {self.unassigned_objective}")
                            print(f"Swaps = {selected_node.swaps_performed}, Weight = {self.swaps_objective}")
                            depth = max(selected_node.hw_qubit_depth)
                            print(f"Depth = {depth}, Weight = {self.depth_objective}")
                            print(f"Score = {selected_node.score_estimate}")
                            print(f"Level solution node = {selected_node.level}")
                            print(f"Nodes added = {self.nodes_added}")
                            print(f"Nodes selected = {self.nodes_selected}")
                            print(f"Execution time = {t_ellapsed} sec")
                        return (selected_node, TerminationCause.SOLUTIONFOUND)

                # Update max values
                max_level = max(max_level, selected_node.level)
                remain_gates_improved = selected_node.nmb_rem_gates < min_remaining_gates
                if remain_gates_improved:
                    min_remaining_gates = selected_node.nmb_rem_gates
                    self.diving_strategy.min_gates_updated(min_remaining_gates)

                # Dive if diving condition is triggered
                if not running_statistics and not -1 in selected_node.qubit_hardware_map and self.diving_strategy.can_dive():
                    dive_solution = self._dive_to_solution(selected_node)
                    improves = self.current_best_solution == None or dive_solution[0] < self.current_best_solution[0]
                    if improves:
                        self.current_best_solution = dive_solution
                    self.diving_strategy.after_dive(improves)
                else:
                    self.diving_strategy.after_non_dive()

                # Expand
                if selected_node.nmb_rem_gates > 0:
                    self._expand_node(selected_node)

                # Log status if it should be done now
                if log and not is_beam and t_now >= t_next_log:
                    if not has_logged:
                        print(f"Totally {total_gates} gates to be placed")
                        print("    Sec     Nodes  Selected   Lev Rem-g Heap Up-B")
                        #print("    Sec     Nodes  Selected   Lev Rem-g Lo-B Up-B")
                        has_logged = True
                    depth_ub = "   NA" if self.current_best_solution == None else "{:5d}".format(self.current_best_solution[0])
                    heap_key = selected_node.score_estimate + self.remain_gates_weight * selected_node.nmb_rem_gates
                    print("{:7.1f}".format(t_ellapsed) + "{:10d}".format(self.nodes_added) + "{:10d}".format(self.nodes_selected) + "{:6d}".format(max_level) + "{:6d}".format(min_remaining_gates) + "{:5d}".format(heap_key) + depth_ub)
                    #print("{:7.1f}".format(t_ellapsed) + "{:10d}".format(self.nodes_added) + "{:10d}".format(self.nodes_selected) + "{:6d}".format(max_level) + "{:6d}".format(min_remaining_gates) + "{:5d}".format(selected_node.score_estimate) + depth_ub)
                    # print(self.diving_strategy.count_report())
                    t_next_log += t_step

        if running_statistics:
            stat_t0 = time.time()
            self.statistics.set_final_score(self.origin, self)
            stat_t1 = time.time()
            self.statistics.print_report()
            print(f"Time set final score: {stat_t1 - stat_t0}")
        elif log:
            print("No solution found!")
        return (None, TerminationCause.EXHAUSTED)
