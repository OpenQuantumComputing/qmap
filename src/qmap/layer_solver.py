import sys
import json
import os
import glob
import getopt
import time
import heapq
from functools import cmp_to_key
from dataclasses import dataclass, field
from problem import Problem
from problem_encoding import ProblemEncoding, GateEncoding
from hardware_distance import HardwareDistance
from solution import Solution, SolutionJob, sol_job_compare


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

    def __init__(self, id : int, parent, job : Job, qubit_hardware_map : list[int], hardware_qubit_map : dict[int, int], hw_qubit_depth : list[int]) -> None:

        self.id = id
        self.level = 0 if parent == None else 1 + parent.level
        self.parent = parent
        self.job = job
        self.qubit_hardware_map = qubit_hardware_map
        self.hardware_qubit_map = hardware_qubit_map
        self.hw_qubit_depth = hw_qubit_depth
        self.in_solution = False

    def __repr__(self):
        return f'id:{self.id},parent:{"-" if self.parent == None else self.parent.id},level:{self.level},job:{self.job},logic_to_hw:{self.qubit_hardware_map}'


# Data class for the priority queue that ignores the second element
@dataclass(order=True)
class PrioritizedItem:

    priority: int
    item: Node=field(compare=False)

class LayerAlgorithm:

    def __init__(self, encoding : ProblemEncoding, hw_distances : HardwareDistance) -> None:

        self.encoding = encoding
        self.problem = encoding.problem
        self.hw_distances = hw_distances

        # Initialize queue
        self.queue = None
        self.nodes_selected = 0
        self.nodes_added = 0
        self.previous_node_id = -1
        self.origin : Node = None
        self.last_completed = encoding.nmb_logical_qubits == 0


    def _add_node(self, node : Node, next_gates : list[GateEncoding]) -> None:

        heap_key = sum(self.hw_distances.swap_distance[gate.pair[0]][gate.pair[1]] for gate in next_gates) + node.level
        heapq.heappush(self.queue, PrioritizedItem(heap_key, node))
        self.nodes_added += 1
        # print(f"Node added: sc={heap_key} node={node}")

    def _select_node(self) -> Node:

        selected_node : Node = heapq.heappop(self.queue).item
        self.nodes_selected += 1

        return selected_node


    def _get_next_node_id(self) -> int:

        self.previous_node_id += 1
        return self.previous_node_id


    def _closest_free_qubits(self, free_qubits : list[int]) -> tuple[int, int]:

        closest_dist = 2 * self.encoding.nmb_hardware_qubits
        closest_pair : tuple[int, int] = None
        for q0 in free_qubits:
            for q1 in free_qubits:
                if q0 < q1:
                    dist = self.hw_distances.swap_distance[q0][q1]
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_pair = (q0, q1)
                        if dist == 0:
                            return closest_pair

        return closest_pair


    def _expand_node(self, node_to_expand : Node, next_gates : list[GateEncoding]):

        for hw_from, hw_to in self.encoding.hardware_edges:
            q_from = -1 if not hw_from in node_to_expand.hardware_qubit_map else node_to_expand.hardware_qubit_map[hw_from]
            q_to = -1 if not hw_to in node_to_expand.hardware_qubit_map else node_to_expand.hardware_qubit_map[hw_to]

            if q_from != -1 or q_to != -1:
                # At least one hardware edge is assigned to a logical qubit, so we can add a swap.

                # Get start time of swap and updated hardware qubit depths.
                hw_qubit_depth = node_to_expand.hw_qubit_depth.copy()
                swap_start = max(hw_qubit_depth[hw_from], hw_qubit_depth[hw_to])
                hw_qubit_depth[hw_from] = hw_qubit_depth[hw_to] = swap_start + self.problem.swap_time

                # Get updated assignment maps of logical and hardware qubits
                hardware_qubit_map = node_to_expand.hardware_qubit_map.copy()
                qubit_hardware_map = node_to_expand.qubit_hardware_map.copy()
                if q_from != -1:
                    qubit_hardware_map[q_from] = hw_to
                    hardware_qubit_map[hw_to] = q_from
                else:
                    del hardware_qubit_map[hw_to]
                if q_to != -1:
                    qubit_hardware_map[q_to] = hw_from
                    hardware_qubit_map[hw_from] = q_to
                else:
                    del hardware_qubit_map[hw_from]

                # Add node for swap operation to queue
                swap_job = Job(None, (hw_from, hw_to), swap_start, self.problem.swap_time)
                node = Node(self._get_next_node_id(), node_to_expand, swap_job, qubit_hardware_map, hardware_qubit_map, hw_qubit_depth)
                self._add_node(node, next_gates)


    def solve(self) -> Node:

        # Initialize logging and timing
        t_start = time.time()
        t_step = 5.0
        t_next_log = t_start + t_step
        t_ellapsed = 0.0
        has_logged = False
        level_start_node = 0
        max_level = 0
        remaining_gates = len(self.encoding.gates)

        # Initialize queue
        self.queue = []
        self.nodes_selected = 0
        self.nodes_added = 0
        self.previous_node_id = -1

        # Next gate for qubit
        next_qubit_gate = [0] * self.encoding.nmb_logical_qubits

        # Original setup of qubit assignments and depths
        qubit_hardware_map = [-1] * self.encoding.nmb_logical_qubits
        hardware_qubit_map : dict[int, int] = {}
        hw_qubit_depth = [0] * self.encoding.nmb_hardware_qubits

        # Find list of next gates that can be planned, and assign their qubits to hardware qubits
        next_gates = [ gate for gate in self.encoding.gates if self.encoding.gate_position_for_qubit[gate.id] == (0, 0)]
        unassigned_hardware_qubits = list(range(self.encoding.nmb_hardware_qubits))
        for gate in next_gates:

            hw_qubit_pair = next((e for e in self.encoding.hardware_edges if e[0] not in hardware_qubit_map and e[1] not in hardware_qubit_map), None)
            if hw_qubit_pair == None:
                closest_dist = 2 * self.encoding.nmb_hardware_qubits
                for q0 in unassigned_hardware_qubits:
                    for q1 in unassigned_hardware_qubits:
                        if q0 < q1:
                            dist = self.hw_distances.swap_distance[q0][q1]
                            if dist < closest_dist:
                                closest_dist = dist
                                hw_qubit_pair = (q0, q1)
            qubit_hardware_map[gate.pair[0]] = hw_qubit_pair[0]
            qubit_hardware_map[gate.pair[1]] = hw_qubit_pair[1]
            hardware_qubit_map[hw_qubit_pair[0]] = gate.pair[0]
            hardware_qubit_map[hw_qubit_pair[1]] = gate.pair[1]
            unassigned_hardware_qubits.remove(hw_qubit_pair[0])
            unassigned_hardware_qubits.remove(hw_qubit_pair[1])

        # Add root node to search tree
        origin = Node(self._get_next_node_id(), None, None, qubit_hardware_map, hardware_qubit_map, hw_qubit_depth)
        self._add_node(origin, next_gates)

        while self.queue:

            # Select
            t_now = time.time()
            current_node = self._select_node()
            # print(f"Select: hardware_qubit_map = {current_node.hardware_qubit_map}")

            # Test if any gates can be planned
            qubit_hardware_map = current_node.qubit_hardware_map
            # print(f"Testing of assignment possibilities for {current_node}")
            # for gate in next_gates:
            #     print(f"    For {gate}, distance({gate.pair[0]}=>{qubit_hardware_map[gate.pair[0]]},{gate.pair[1]}=>{qubit_hardware_map[gate.pair[1]]}) = {self.hw_distances.swap_distance[qubit_hardware_map[gate.pair[0]]][qubit_hardware_map[gate.pair[1]]]}")
            if any(self.hw_distances.swap_distance[qubit_hardware_map[gate.pair[0]]][qubit_hardware_map[gate.pair[1]]] == 0 for gate in next_gates):

                # Some gates can be planned. Plan all that can be planned from the current qubit assignment, then continue algorithm from that state.
                try_add_gates = True
                new_qubits_assigned = False
                while try_add_gates:

                    # Some of the next plannable gates that can be planned
                    try_add_gates = False
                    for gate in next_gates:

                        hw_0 = qubit_hardware_map[gate.pair[0]]
                        hw_1 = qubit_hardware_map[gate.pair[1]]
                        # print(f"Testing if {gate} can be added: Dist({hw_0},{hw_1}) = {self.hw_distances.swap_distance[hw_0][hw_1]}")
                        if self.hw_distances.swap_distance[hw_0][hw_1] == 0:

                            hw_qubit_depth = current_node.hw_qubit_depth.copy()
                            start_time = max(hw_qubit_depth[hw_0], hw_qubit_depth[hw_1])
                            hw_qubit_depth[hw_0] = hw_qubit_depth[hw_1] = start_time + gate.duration
                            job = Job(gate, (hw_0, hw_1), start_time, gate.duration)

                            current_node = Node(self._get_next_node_id(), current_node, job, current_node.qubit_hardware_map, current_node.hardware_qubit_map, hw_qubit_depth)

                            next_qubit_gate[gate.pair[0]] += 1
                            next_qubit_gate[gate.pair[1]] += 1

                            try_add_gates = True

                    # Update list of next plannable gates
                    # print(f"Try_add = {try_add_gates}  Old next_gates = {next_gates}")
                    if try_add_gates:
                        next_gates = []
                        for q0 in range(self.encoding.nmb_logical_qubits):
                            if next_qubit_gate[q0] < self.encoding.nmb_qubit_gates[q0]:
                                gate = self.encoding.qubit_gates[q0][next_qubit_gate[q0]]
                                q1 = gate.pair[1]
                                if q0 == gate.pair[0] and next_qubit_gate[q1] == self.encoding.gate_position_for_qubit[gate.id][1]:

                                    # Found a next plannable gate
                                    next_gates.append(gate)

                                    # Update qubit assignment if one of the logical qubits are not assigned. We then know the other logical qubit is assigned.
                                    if qubit_hardware_map[q0] == -1 or qubit_hardware_map[q1] == -1:

                                        # print(f"new_qubits_assigned = {new_qubits_assigned}")
                                        if not new_qubits_assigned:

                                            # This is the first time since the last swap node was added that a gate with missing assignment for one of its qubits
                                            qubit_hardware_map = current_node.qubit_hardware_map.copy()
                                            hardware_qubit_map = current_node.hardware_qubit_map.copy()
                                            unassigned_hardware_qubits = [q for q in range(self.encoding.nmb_hardware_qubits) if not q in hardware_qubit_map]
                                            # print(f"qubit_hardware_map = {qubit_hardware_map}")
                                            # print(f"hardware_qubit_map = {hardware_qubit_map}")
                                            # print(f"unassigned_hardware_qubits = {unassigned_hardware_qubits}")
                                            new_qubits_assigned = True

                                        if qubit_hardware_map[q0] == -1:
                                            log_unassigned = q0
                                            hw_assigned = qubit_hardware_map[q1]
                                        else:
                                            log_unassigned = q1
                                            hw_assigned = qubit_hardware_map[q0]

                                        # Update assignments to include both gate qubits
                                        hw_unassigned = min(unassigned_hardware_qubits, key = lambda q: self.hw_distances.swap_distance[hw_assigned][q])
                                        qubit_hardware_map[log_unassigned] = hw_unassigned
                                        hardware_qubit_map[hw_unassigned] = log_unassigned
                                        unassigned_hardware_qubits.remove(hw_unassigned)
                        # print(f"New next_gates = {next_gates}")

                # No more gates can be assigned. Update qubit assignment maps for node if that has changed
                if new_qubits_assigned:
                    current_node.qubit_hardware_map = qubit_hardware_map
                    current_node.hardware_qubit_map = hardware_qubit_map

                # Mark nodes that are part of solution
                n = current_node
                while n != None and not n.in_solution:
                    n.in_solution = True
                    n = n.parent

                # Remove all nodes from queue and remove parent information for nodes not in solution, to help garbage collection
                for n_item in self.queue:
                    n : Node = n_item.item
                    while n != None and not n.in_solution:
                        n2 = n.parent
                        n.parent = None
                        n = n2
                self.queue = []

                # If there are still gates to be planned, continue the search from the current node. If not, solution is found
                twice_remaining_gates = sum(self.encoding.nmb_qubit_gates[q] - next_qubit_gate[q] for q in range(self.encoding.nmb_logical_qubits))
                if next_gates:
                    if twice_remaining_gates <= 0:
                        raise Exception("Unexpected remaining gates found")
                    self._add_node(current_node, next_gates)
                    level_start_node = current_node.level
                    max_level = level_start_node
                    remaining_gates = twice_remaining_gates // 2
                else:
                    if twice_remaining_gates != 0:
                        raise Exception("Expected to find some next gates to be planned")

                    print(f"Solution found")
                    print(f"Nodes added = {self.nodes_added}")
                    print(f"Nodes selected = {self.nodes_selected}")
                    duration = time.time() - t_start
                    print(f"Execution time = {duration} sec")

                    return current_node

            else:

                # Expand
                self._expand_node(current_node, next_gates)

                # Log status if it should be done now
                max_level = max(max_level, current_node.level)
                if t_now >= t_next_log:
                    if not has_logged:
                        print(f"Totally {len(self.encoding.gates)} gates to be placed")
                        print("    Sec     Nodes  Selected   Lev Rem-g")
                        has_logged = True
                    t_ellapsed += t_step
                    print("{:7.1f}".format(t_ellapsed) + "{:10d}".format(self.nodes_added) + "{:10d}".format(self.nodes_selected) + "{:6d}".format(max_level - level_start_node) + "{:6d}".format(remaining_gates))
                    t_next_log += t_step


def make_solution(end_node : Node, encoding : ProblemEncoding) -> Solution:

    if end_node == None:
        return None

    hardware_to_qubit: dict[int, int] = end_node.hardware_qubit_map.copy()
    n = end_node
    sol_jobs : list[SolutionJob] = []

    while n != None:
        job = n.job
        if job != None:
            if job.gate == None:
                hw_from = job.hardware_pair[0]
                hw_to = job.hardware_pair[1]
                q_from = hardware_to_qubit[hw_to] if hw_to in hardware_to_qubit else -1
                q_to = hardware_to_qubit[hw_from] if hw_from in hardware_to_qubit else -1
                if q_from == -1:
                    hardware_to_qubit.pop(hw_from)
                else:
                    hardware_to_qubit[hw_from] = q_from
                if q_to == -1:
                    hardware_to_qubit.pop(hw_to)
                else:
                    hardware_to_qubit[hw_to] = q_to
                prob_gate = None
            else:
                hw_from_idx = 1 if job.hardware_pair[0] in hardware_to_qubit and hardware_to_qubit[job.hardware_pair[0]] == job.gate.pair[1] else 0
                hw_from = job.hardware_pair[hw_from_idx]
                hw_to = job.hardware_pair[1 - hw_from_idx]
                prob_gate = encoding.problem.gates[job.gate.id]
            sol_jobs.append(SolutionJob(job.start, job.duration, prob_gate, (encoding.hardware_qubit_names[hw_from], encoding.hardware_qubit_names[hw_to])))
        n = n.parent
    
    sol_jobs.sort(key = cmp_to_key(sol_job_compare))
    hardware_to_qubit_names = {encoding.hardware_qubit_names[hw_q]: encoding.logical_qubit_names[lo_q] for hw_q, lo_q in hardware_to_qubit.items()}
    return Solution(encoding.problem, False, hardware_to_qubit_names, sol_jobs)


def solve_single_problem(file_path_in : str, file_path_out : str) -> tuple[int, int, int]:

    # Create problem
    with open(file_path_in, "r") as input_file:
        data = json.load(input_file)
    print(f"Input read from {file_path_in}")
    problem_json = data["problem"]
    problem = Problem(problem_json)
    # Use next line only to get number of gates output
    #return (len(problem.gates), 10, 10)

    # Create algorithm
    encoding = ProblemEncoding(problem)
    hw_distances = HardwareDistance(encoding, False)
    algorithm = LayerAlgorithm(encoding, hw_distances)

    # Solve problem
    solution_node = algorithm.solve()
    solution = make_solution(solution_node, encoding)
    solution_depth = -1
    solution_swaps = -1

    if solution:
        validation_error = solution.validate(problem.swap_time)
        if validation_error:
            print(f"Layer algorithm solution validation failed: {validation_error}")
        else:
            solution_depth = solution.depth()
            solution_swaps = solution.swaps()
            print(f"Solution depth: {solution_depth}")
            print(f"Swaps in solution: {solution_swaps}")

            if file_path_out != None:
                solution_json = solution.to_json_dict()
                dynamic_graph_solution = {"solution": solution_json}
                data["dynamic_graph_solution"] = dynamic_graph_solution
                json_object = json.dumps(data)
                # json_object = json.dumps(data, indent=4)
                with open(file_path_out, "w") as outfile:
                    outfile.write(json_object)
                print(f"Solution added to input data and stored to {file_path_out}")

    return (len(problem.gates), solution_depth, solution_swaps)


def solve_problems(problem_files : list[tuple[str, str]]) -> None:

    log_lines : list[str] = []
    for file_path_in, file_path_out in problem_files:
        t0 = time.time()
        nmb_gates, sol_depth, sol_swaps = solve_single_problem(file_path_in, file_path_out)
        t1 = time.time()
        if sol_depth == -1:
            log_lines.append(f"{file_path_in} gates={nmb_gates} time={t1-t0}   *** NO SOLUTION FOUND ***")
        else:
            log_lines.append(f"{file_path_in} gates={nmb_gates} depth={sol_depth} swaps={sol_swaps} time={t1-t0}")
    print()
    for l in log_lines:
        print(l)

if __name__=="__main__":

    opt_arguments = sys.argv[1:]

    single_file_in = None
    single_file_out = None
    input_dir = None
    output_dir = None
    reg_exp = None

    options = "I:i:O:o:r:"
    long_options = ["input_directory=", "input_file=", "output_directory=", "output_file=", "reg_exp="]

    try:
        arguments, values = getopt.getopt(opt_arguments, options, long_options)

        for argument, value in arguments:

            if argument in ("-I", "--input_directory"):
                input_dir = value.replace("\\","/")
            elif argument in ("-i", "--input_file"):
                single_file_in = value.replace("\\","/")
            elif argument in ("-O", "--output_directory"):
                output_dir = value.replace("\\","/")
            elif argument in ("-o", "--output_file"):
                single_file_out = value.replace("\\","/")
            elif argument in ("-r", "--reg_exp"):
                reg_exp = value.replace("\\","/")

        problem_files : list[tuple[str, str]] = []
        if single_file_in != None:
            problem_files.append((single_file_in, single_file_out))
        if input_dir != None:
            if reg_exp == None:
                print("Can not run for all files in directory if not reg_exp is set")
                exit(1)
            else:
                search_path = os.path.join(input_dir, reg_exp)
                files = glob.glob(search_path)
                for file_path_in in files:
                    file_path_in = file_path_in.replace("\\", "/")
                    if output_dir == None:
                        file_path_out = None
                    else:
                        file_path_out = file_path_in.replace(input_dir, output_dir)
                    problem_files.append((file_path_in, file_path_out))

        solve_problems(problem_files)

    except getopt.error as err:
        print(str(err))
