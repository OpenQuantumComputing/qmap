from score_estimation import DepthEstimator
from problem_encoding import ProblemEncoding
from hardware_distance import HardwareDistance

class GateDependencyDepths:
    """Gives some minimum depth differences from gates to end under complete hardware graphs assumptions.
    (1) Gives, for each logical qubit and each gate for that qubit, the minimum difference between when the entire circuit is completed and when the execution of the given gate is started.
    (2) Gives, for each logical qubit and each gate for that qubit, the total durations of the remaining gates of the qubit, starting with the given gate.
    (3) Gives, for each logical qubit and each gate for that qubit, the minimum total time when no gate of the qubit is executed until the circuit is completed.
    For both of these numbers, the calculation is done for a complete hardware graph under the assumption that only the given gate and all gates depending on it are the remaining gates not executed yet."""


    encoding : ProblemEncoding
    """The encoding of the input problem."""

    least_depth_from_gate : list[list[int]]
    """For each logical qubit and each index among all gates for that qubit, the minimum difference between when the entire circuit is completed and when the execution of the given gate is started.
    If the index is the number of gates for the qubit, 0 is returned.
    The number is given under the assumption that the hardware graph is complete and only the given gate and all gates depending on it are the remaining gates not executed yet."""

    durations_from_gate : list[list[int]]
    """For each logical qubit and each gate for that qubit, the total durations of the remaining gates of the qubit, starting with the given gate.
    If the index is the number of gates for the qubit, 0 is returned.
    The number is given under the assumption that the hardware graph is complete and only the given gate and all gates depending on it are the remaining gates not executed yet."""

    least_free_from_gate : list[list[int]]
    """For each logical qubit and each index among all gates for that qubit, the minimum total time when no gate of the qubit is executed until the circuit is completed.
    If the index is the number of gates for the qubit, 0 is returned.
    The number is given under the assumption that the hardware graph is complete and only the given gate and all gates depending on it are the remaining gates not executed yet."""


    def __init__(self, encoding : ProblemEncoding) -> None:

        self.encoding = encoding
        self.least_depth_from_gate: list[list[int]] = list([0] * (1 + nmb_gates) for nmb_gates in encoding.nmb_qubit_gates)
        self.durations_from_gate: list[list[int]] = list([0] * (1 + nmb_gates) for nmb_gates in encoding.nmb_qubit_gates)
        self._build_least_gate_depth_()
        self.least_free_from_gate: list[list[int]] = list(list(self.least_depth_from_gate[i][j] - self.durations_from_gate[i][j] for j in range(len(self.least_depth_from_gate[i]))) for i in range(self.encoding.nmb_logical_qubits))

    def _build_least_gate_depth_(self):

        for gate in reversed(self.encoding.gates):
            gate_pos = self.encoding.gate_position_for_qubit[gate.id]
            max_succ_depth = max(self.least_depth_from_gate[q][pos + 1] for (q, pos) in ((gate.pair[index], gate_pos[index]) for index in range(2)))
            depth = max_succ_depth + gate.duration
            for index in range(2):
                self.least_depth_from_gate[gate.pair[index]][gate_pos[index]] = depth
                self.durations_from_gate[gate.pair[index]][gate_pos[index]] = self.durations_from_gate[gate.pair[index]][gate_pos[index] + 1] + gate.duration


class LeastSwapsDepthEstimator(DepthEstimator):

    def __init__(self, gate_depths : GateDependencyDepths, hw_distances : HardwareDistance) -> None:

        super().__init__()
        self.gate_depths = gate_depths
        self.encoding = gate_depths.encoding
        self.swap_time = gate_depths.encoding.problem.swap_time
        self.hw_distances = hw_distances

    def depth_estimate(self, hw_qubit_depth: list[int], next_qubit_gate: list[int], qubit_hardware_map: list[int], hardware_qubit_map: list[int]) -> int:

        # First get least depth needed based on current depth and remaining gates for each qubit, assuming a complete hardware graph
        qubit_complete_depth = [(0 if next_gate_q == self.encoding.nmb_qubit_gates[qubit] else self.gate_depths.least_depth_from_gate[qubit][next_gate_q]) + (0 if qubit_hardware_map[qubit] == -1 else hw_qubit_depth[qubit_hardware_map[qubit]]) for qubit, next_gate_q in enumerate(next_qubit_gate)]
        max_depth = max(qubit_complete_depth)

        # Next run through all remaining gates and extend depth by needed swaps to bring the qubits together. Only needed for gates where both qubits are assigned to hardware positions that are not neighbours
        qubit = 0
        for hw_q in qubit_hardware_map:
            if hw_q != -1:
                # Qubit is assigned to hardware, iterate through gates where 'qubit' is first gate
                free_q_base = max_depth - qubit_complete_depth[qubit] + self.gate_depths.least_free_from_gate[qubit][next_qubit_gate[qubit]]
                for gate_index_q in range(next_qubit_gate[qubit], self.encoding.nmb_qubit_gates[qubit]):
                    gate = self.encoding.qubit_gates[qubit][gate_index_q]
                    if qubit == gate.pair[0]:
                        qubit_other = gate.pair[1]
                        hw_other = qubit_hardware_map[qubit_other]
                        if hw_other != -1:
                            needed_swaps = self.hw_distances.swap_distance[hw_q][hw_other]
                            if needed_swaps > 0:
                                # Other qubit in gate is also assigned to hardware, and the qubits are not hardware neighbours
                                gate_pos_other = self.encoding.gate_position_for_qubit[gate.id][1]
                                free_q = free_q_base - self.gate_depths.least_free_from_gate[qubit][gate_index_q]
                                free_swaps = free_q // self.swap_time
                                free_q_other = max_depth - qubit_complete_depth[qubit_other] + self.gate_depths.least_free_from_gate[qubit_other][next_qubit_gate[qubit_other]] - self.gate_depths.least_free_from_gate[qubit_other][gate_pos_other]
                                free_swaps_other = free_q_other // self.swap_time
                                needed_swaps -= free_swaps + free_swaps_other
                                if needed_swaps > 0:
                                    # There is not enough space for all needed swaps, extend depth
                                    remain = free_q - free_swaps * self.swap_time
                                    remain_other = free_q_other - free_swaps_other * self.swap_time
                                    if (needed_swaps % 2) == 0:
                                        max_depth_add = (needed_swaps // 2) * self.swap_time - min(remain, remain_other)
                                    else:
                                        max_depth_add = ((needed_swaps + 1) // 2) * self.swap_time - max(remain, remain_other)
                                    max_depth += max_depth_add
                                    free_q_base += max_depth_add
            qubit += 1

        return max_depth


class PathSwapsDepthEstimator(DepthEstimator):

    def __init__(self, gate_depths : GateDependencyDepths, hw_distances : HardwareDistance) -> None:

        super().__init__()
        self.gate_depths = gate_depths
        self.encoding = gate_depths.encoding
        self.swap_time = gate_depths.encoding.problem.swap_time
        self.hw_distances = hw_distances


    def depth_estimate(self, hw_qubit_depth: list[int], next_qubit_gate: list[int], qubit_hardware_map: list[int], hardware_qubit_map: list[int]) -> int:

        # First get least depth needed based on current depth and remaining gates for each qubit, assuming a complete hardware graph
        qubit_complete_depth = [(0 if next_gate_q == self.encoding.nmb_qubit_gates[qubit] else self.gate_depths.least_depth_from_gate[qubit][next_gate_q]) + (0 if qubit_hardware_map[qubit] == -1 else hw_qubit_depth[qubit_hardware_map[qubit]]) for qubit, next_gate_q in enumerate(next_qubit_gate)]
        max_depth = max(qubit_complete_depth)

        # Next run through all remaining gates and extend depth by needed swaps to bring the qubits together. Only needed for gates where both qubits are assigned to hardware positions that are not neighbours
        qubit = 0
        for hw_q in qubit_hardware_map:
            if hw_q != -1:
                # Qubit is assigned to hardware, iterate through gates where 'qubit' is first gate
                free_q_base = max_depth - qubit_complete_depth[qubit] + self.gate_depths.least_free_from_gate[qubit][next_qubit_gate[qubit]]
                for gate_index_q in range(next_qubit_gate[qubit], self.encoding.nmb_qubit_gates[qubit]):
                    gate = self.encoding.qubit_gates[qubit][gate_index_q]
                    if qubit == gate.pair[0]:
                        qubit_other = gate.pair[1]
                        hw_other = qubit_hardware_map[qubit_other]
                        if hw_other != -1:
                            needed_swaps = self.hw_distances.swap_distance[hw_q][hw_other]
                            if needed_swaps > 0:
                                # Other qubit in gate is also assigned to hardware, and the qubits are not hardware neighbours
                                gate_pos_other = self.encoding.gate_position_for_qubit[gate.id][1]
                                for path in self.hw_distances.hardware_paths[hw_q][hw_other]:
                                    free_0 = free_q_base - self.gate_depths.least_free_from_gate[qubit][gate_index_q]
                                    free_1 = max_depth - qubit_complete_depth[qubit_other] + self.gate_depths.least_free_from_gate[qubit_other][next_qubit_gate[qubit_other]] - self.gate_depths.least_free_from_gate[qubit_other][gate_pos_other]
                                    max_depth_add = 0
                                    pos_0 = 0
                                    pos_1 = len(path) - 1
                                    depth_0 = max_depth - free_0 - self.gate_depths.least_depth_from_gate[qubit][gate_index_q]
                                    depth_1 = max_depth - free_1 - self.gate_depths.least_depth_from_gate[qubit_other][gate_pos_other]
                                    new_depth_0 = max(depth_0, hw_qubit_depth[path[pos_0 + 1]]) + self.swap_time
                                    new_depth_1 = max(depth_1, hw_qubit_depth[path[pos_1 - 1]]) + self.swap_time
                                    depth_add_0 = new_depth_0 - depth_0 - free_0
                                    depth_add_1 = new_depth_1 - depth_1 - free_1
                                    while pos_0 + 2 <= pos_1:
                                        if depth_add_0 <= depth_add_1:
                                            depth_0 = new_depth_0
                                            pos_0 += 1
                                            if depth_add_0 <= 0:
                                                free_0 = -depth_add_0
                                            else:
                                                free_0 = 0
                                                free_1 += depth_add_0
                                                depth_add_1 -= depth_add_0
                                                max_depth_add += depth_add_0
                                            new_depth_0 = max(depth_0, hw_qubit_depth[path[pos_0 + 1]]) + self.swap_time
                                            depth_add_0 = new_depth_0 - depth_0 - free_0
                                        else:
                                            depth_1 = new_depth_1
                                            pos_1 -= 1
                                            if depth_add_1 <= 0:
                                                free_1 = -depth_add_1
                                            else:
                                                free_1 = 0
                                                free_0 += depth_add_1
                                                depth_add_0 -= depth_add_1
                                                max_depth_add += depth_add_1
                                            new_depth_1 = max(depth_1, hw_qubit_depth[path[pos_1 - 1]]) + self.swap_time
                                            depth_add_1 = new_depth_1 - depth_1 - free_1
                                    max_depth += max_depth_add
                                    free_q_base += max_depth_add
            qubit += 1

        return max_depth


class IndexedPathSwapsDepthEstimator(DepthEstimator):
    """An admissible heuristic of the final depth needed to complete from a given node state in the dynamic algorithm, based on swap applications along paths connecting the qubits in a remaining gate.
    For a remaining gate, an optimistic measure is done for each of the gate qubits on how early the qubit is ready to execute the gate, and how early all gates can be completed after the completion of the given gate.
    This measure takes into account the current depth of the qubit and the gates to be executed for that qubit before the given gate.
    In addition, if the qubits of the gate are not assigned to two neighbour hardware qubits, all different paths connecting the qubits are tested to see how fast the qubits can be swapped towards each other.
    This takes into account the position of the other qubits in the swap operations.
    The returned measure is the maximum taken over all such gates.
    """


    gate_depths : GateDependencyDepths
    """Calculation of minimum remaining times and free periods from a given gate to the completion of all gates."""

    encoding : ProblemEncoding
    """The encoding of the input problem."""

    swap_time : int
    """The time spent by a swap operation."""

    hw_distances : HardwareDistance
    """Hardware distances and minimal paths of length >= 3 in the hardware graph. A minimal path is a path that can not be shortened by removing some inner nodes."""

    first_shared_gate: list[list[list[int]]]
    """Gives the first unplanned gate shared by two given logical qubits, based on a current state in the planning process.
    For two qubits q1, q2 where 0 <= q1 < q2 < number of logical qubits, and for a gate index g where 0 <= g <= G and G is the number of gates for q2,
    first_shared_gate[q2][g][q1] is the index of the first gate among the gates for q2 that is not planned yet if g < G and any such gate exists.
    I.e. if such gates exist, first_shared_gate[q2][g][q1] is the smallest number i >= g such that gate number i amoung the gates for q2 is a common gate for q1 and q2.
    if f = G or there is no such gate, first_shared_gate[q2][g][q1] = G.
    first_shared_gate[0] is None since the variable is only used for cases when q2 >= 1.
    """

    def __init__(self, gate_depths : GateDependencyDepths, hw_distances : HardwareDistance) -> None:

        super().__init__()
        self.gate_depths = gate_depths
        self.encoding = gate_depths.encoding
        self.swap_time = gate_depths.encoding.problem.swap_time
        self.hw_distances = hw_distances
        self.first_shared_gate: list[list[list[int]]] = list(self._initial_first_shared_gate(idx) for idx in range(self.encoding.nmb_logical_qubits))
        self._build_first_shared_gate()


    def _initial_first_shared_gate(self, idx : int) -> list[list[int]]:
        if idx == 0:
            return None
        else:
            gates = self.encoding.nmb_qubit_gates[idx]
            return list([gates] * idx if pos == gates else [] for pos in range(1 + gates))


    def _build_first_shared_gate(self) -> None:

        for gate in reversed(self.encoding.gates):
            gate_pos = self.encoding.gate_position_for_qubit[gate.id]
            if gate.pair[0] < gate.pair[1]:
                q_min = gate.pair[0]
                q_max = gate.pair[1]
                pos_min = gate_pos[0]
                pos_max = gate_pos[1]
            else:
                q_min = gate.pair[1]
                q_max = gate.pair[0]
                pos_min = gate_pos[1]
                pos_max = gate_pos[0]

            if q_min > 0:
                self.first_shared_gate[q_min][pos_min] = self.first_shared_gate[q_min][pos_min + 1].copy()
            self.first_shared_gate[q_max][pos_max] = self.first_shared_gate[q_max][pos_max + 1].copy()
            self.first_shared_gate[q_max][pos_max][q_min] = pos_max


    def depth_estimate(self, hw_qubit_depth: list[int], next_qubit_gate: list[int], qubit_hardware_map: list[int], hardware_qubit_map: list[int]) -> int:

        if -1 in qubit_hardware_map:
            return self.depth_estimate_with_unassigned(hw_qubit_depth, next_qubit_gate, qubit_hardware_map, hardware_qubit_map)

        # First get least depth needed based on current depth and remaining gates for each qubit, assuming a complete hardware graph.
        qubit_complete_depth = [(0 if next_gate_q == self.encoding.nmb_qubit_gates[qubit] else self.gate_depths.least_depth_from_gate[qubit][next_gate_q]) + hw_qubit_depth[qubit_hardware_map[qubit]] for qubit, next_gate_q in enumerate(next_qubit_gate)]
        max_depth = max(qubit_complete_depth)

        # Next run through all remaining gates and extend depth by needed swaps to bring the qubits together. Only needed for gates where the two qubits are assigned to hardware positions that are not neighbours.
        for q_max in range(1, self.encoding.nmb_logical_qubits):
            hw_max = qubit_hardware_map[q_max]
            # Run through gates where the highest qubit index is q_max, and where hw_max is the hardware qubit q_max is currently assigned to.

            first_shared_max = self.first_shared_gate[q_max][next_qubit_gate[q_max]]
            for q_min in range(0, q_max):
                hw_min = qubit_hardware_map[q_min]
                gate_pos_max = first_shared_max[q_min]
                if gate_pos_max < self.encoding.nmb_qubit_gates[q_max] and self.hw_distances.swap_distance[hw_max][hw_min] > 0:

                    # q_min and q_max are logical qubits that are not neighbours and have at least one unplanned common gate, q_min < q_max.
                    # 'gate' will be the first such gate.
                    # gate_pos_max is the index of 'gate' among all gates for q_max.
                    # gate_pos_min is the index of 'gate' among all gates for q_min.
                    gate = self.encoding.qubit_gates[q_max][gate_pos_max]
                    gate_pos_min = self.encoding.gate_position_for_qubit[gate.id][0 if gate.pair[0] == q_min else 1]

                    # For each path connecting the two hardware qubits of the logical qubits in 'gate', we calculate the depth needed to solve the entire problem via the input state,
                    # assuming that the logical qubits must become neighbours by travelling over the path, and by assuming that all other gates can be executed as if the hardware graph was complete.
                    # We assume the free time the qubits must wait for other gates to be completed always can be used on necessary swaps.
                    # 'best_path_free' is the minimum of these depths needed, or -1 if no path has been tested yet. When a path is found with an estimate that is not worse than 'max_depth',
                    # there is no point continuing the search among the paths for this gate.
                    best_path_depth = -1

                    start_gate_0 = qubit_complete_depth[q_max] - self.gate_depths.least_free_from_gate[q_max][next_qubit_gate[q_max]] - self.gate_depths.durations_from_gate[q_max][gate_pos_max]
                    start_gate_1 = qubit_complete_depth[q_min] - self.gate_depths.least_free_from_gate[q_min][next_qubit_gate[q_min]] - self.gate_depths.durations_from_gate[q_min][gate_pos_min]
                    for path in self.hw_distances.hardware_paths[hw_max][hw_min]:
                        if best_path_depth == -1 or best_path_depth > max_depth:

                            path_start_gate_0 = start_gate_0
                            path_start_gate_1 = start_gate_1
                            pos_0 = 0
                            pos_1 = len(path) - 1
                            new_start_gate_0 = max(path_start_gate_0, hw_qubit_depth[path[pos_0 + 1]]) + self.swap_time
                            new_start_gate_1 = max(path_start_gate_1, hw_qubit_depth[path[pos_1 - 1]]) + self.swap_time
                            while pos_0 + 2 <= pos_1:
                                if new_start_gate_0 <= new_start_gate_1:
                                    path_start_gate_0 = new_start_gate_0
                                    pos_0 += 1
                                    new_start_gate_0 = max(path_start_gate_0, hw_qubit_depth[path[pos_0 + 1]]) + self.swap_time
                                else:
                                    path_start_gate_1 = new_start_gate_1
                                    pos_1 -= 1
                                    new_start_gate_1 = max(path_start_gate_1, hw_qubit_depth[path[pos_1 - 1]]) + self.swap_time
                            end_from_gate = max(path_start_gate_0, path_start_gate_1) + self.gate_depths.least_depth_from_gate[q_min][gate_pos_min]
                            new_max_depth = max(max_depth, end_from_gate)
                            if best_path_depth == -1 or new_max_depth < best_path_depth:
                                best_path_depth = new_max_depth

                    if best_path_depth == -1:
                        raise Exception("No depth estimate for path found")

                    if best_path_depth > max_depth:
                        max_depth = best_path_depth

        return max_depth


    def depth_estimate_with_unassigned(self, hw_qubit_depth: list[int], next_qubit_gate: list[int], qubit_hardware_map: list[int], hardware_qubit_map: list[int]) -> int:

        # First get least depth needed based on current depth and remaining gates for each qubit, assuming a complete hardware graph.
        qubit_complete_depth = [(0 if next_gate_q == self.encoding.nmb_qubit_gates[qubit] else self.gate_depths.least_depth_from_gate[qubit][next_gate_q]) + (0 if qubit_hardware_map[qubit] == -1 else hw_qubit_depth[qubit_hardware_map[qubit]]) for qubit, next_gate_q in enumerate(next_qubit_gate)]
        max_depth = max(qubit_complete_depth)

        # Add depths of unassigned hardware qubits
        enumerate(hardware_qubit_map)
        unassigned_hardware_qubits = [hw for hw, qubit in enumerate(hardware_qubit_map) if qubit == -1]
        qubit_hardware_candidates = [unassigned_hardware_qubits if qubit_hardware_map[qubit] == -1 else [qubit_hardware_map[qubit]] for qubit in range(self.encoding.nmb_logical_qubits)]

        unassigned_hardware_qubit_depths = sorted(hw_qubit_depth[qubit] for qubit in unassigned_hardware_qubits)
        unassigned_qubits_remain_depths = sorted(0 if next_gate_q == self.encoding.nmb_qubit_gates[qubit] else self.gate_depths.least_depth_from_gate[qubit][next_gate_q] for qubit, next_gate_q in enumerate(next_qubit_gate) if qubit_hardware_map[qubit] == -1)
        nmb_unas = len(unassigned_qubits_remain_depths)
        max_depth = max(max_depth, max(unassigned_qubits_remain_depths[nmb_unas - 1 - idx] + unassigned_hardware_qubit_depths[idx] for idx in range(nmb_unas)))
        
        # Next run through all remaining gates and extend depth by needed swaps to bring the qubits together. If any of the gate qubits are unassigned, try all positions to assign the qubit at.
        # Only needed for gates that are the first common unplanned gate of its qubits, and where both qubits are not/can not be assigned to hardware positions that are neighbours.
        # First run through selections of highest indexed logical qubit of the gate
        for q_max in range(1, self.encoding.nmb_logical_qubits):

            hws_max = qubit_hardware_candidates[q_max]
            first_shared_max = self.first_shared_gate[q_max][next_qubit_gate[q_max]]
            nmb_gates_max = self.encoding.nmb_qubit_gates[q_max]

            # Run through selections of lowest indexed logical qubit of the gate
            for q_min in range(0, q_max):

                hws_min = qubit_hardware_candidates[q_min]
                gate_pos_max = first_shared_max[q_min]
                if gate_pos_max < nmb_gates_max and all(all(hw_min == hw_max or self.hw_distances.swap_distance[hw_max][hw_min] > 0 for hw_min in hws_min) for hw_max in hws_max):

                    # q_min and q_max are logical qubits that are not/can not be neighbours and have at least one unplanned common gate, q_min < q_max.
                    # 'gate' will be the first such gate.
                    # gate_pos_max is the index of 'gate' among all gates for q_max.
                    # gate_pos_min is the index of 'gate' among all gates for q_min.
                    gate = self.encoding.qubit_gates[q_max][gate_pos_max]
                    gate_pos_min = self.encoding.gate_position_for_qubit[gate.id][0 if gate.pair[0] == q_min else 1]

                    # For all ways to position the logical qubits of 'gate' in the hardware qubits (several choices only for unassigned qubits) and for each path connecting the hardware qubits,
                    # we calculate the depth needed to solve the entire problem via the input state, assuming that the logical qubits must become neighbours by travelling over the path,
                    # and by assuming that all other gates can be executed as if the hardware graph was complete.
                    # We assume the free time the qubits must wait for other gates to be completed always can be used on necessary swaps.
                    # 'best_path_free' is the minimum of these depths needed, or -1 if no path has been tested yet. When a path is found with an estimate that is not worse than 'max_depth',
                    # there is no point continuing the search among the paths for this gate.
                    best_path_depth = -1
                    may_extend_max = True
                    # may_extend_max == best_path_depth == -1 or best_path_depth > max_depth

                    start_gate_0 = qubit_complete_depth[q_max] - self.gate_depths.least_free_from_gate[q_max][next_qubit_gate[q_max]] - self.gate_depths.durations_from_gate[q_max][gate_pos_max]
                    start_gate_1 = qubit_complete_depth[q_min] - self.gate_depths.least_free_from_gate[q_min][next_qubit_gate[q_min]] - self.gate_depths.durations_from_gate[q_min][gate_pos_min]

                    for hw_max in hws_max:
                        if may_extend_max:
                            for hw_min in hws_min:
                                if hw_min != hw_max and may_extend_max:
                                    for path in self.hw_distances.hardware_paths[hw_max][hw_min]:

                                        # We have selected a hardware position for both qubits and a path connecting them, and all depth limits (if any) found are bigger than 'max_depth''
                                        path_start_gate_0 = start_gate_0
                                        path_start_gate_1 = start_gate_1
                                        pos_0 = 0
                                        pos_1 = len(path) - 1
                                        new_start_gate_0 = max(path_start_gate_0, hw_qubit_depth[path[pos_0 + 1]]) + self.swap_time
                                        new_start_gate_1 = max(path_start_gate_1, hw_qubit_depth[path[pos_1 - 1]]) + self.swap_time
                                        while pos_0 + 2 <= pos_1:
                                            if new_start_gate_0 <= new_start_gate_1:
                                                path_start_gate_0 = new_start_gate_0
                                                pos_0 += 1
                                                new_start_gate_0 = max(path_start_gate_0, hw_qubit_depth[path[pos_0 + 1]]) + self.swap_time
                                            else:
                                                path_start_gate_1 = new_start_gate_1
                                                pos_1 -= 1
                                                new_start_gate_1 = max(path_start_gate_1, hw_qubit_depth[path[pos_1 - 1]]) + self.swap_time
                                        end_from_gate = max(path_start_gate_0, path_start_gate_1) + self.gate_depths.least_depth_from_gate[q_min][gate_pos_min]
                                        new_max_depth = max(max_depth, end_from_gate)
                                        if best_path_depth == -1 or new_max_depth < best_path_depth:
                                            best_path_depth = new_max_depth
                                            may_extend_max = best_path_depth > max_depth

                    if best_path_depth == -1:
                        raise Exception("No depth estimate for path found")

                    if best_path_depth > max_depth:
                        max_depth = best_path_depth

        return max_depth
