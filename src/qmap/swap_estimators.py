from hardware_distance import HardwareDistance
from score_estimation import RemainingSwapsEstimator


class AllGatesQubitDistanceSwapEstimator(RemainingSwapsEstimator):

    def __init__(self, hw_distances : HardwareDistance) -> None:

        super().__init__(hw_distances)
        self.encoding = hw_distances.encoding


    def remain_swaps_estimate(self, next_qubit_gate: list[int], qubit_hardware_map: list[int]) -> int:

        max_hardware_dist = 0
        qubit = 0
        for hw_qubit in  qubit_hardware_map:
            if hw_qubit != -1:
                for gate in self.encoding.qubit_gates[qubit][next_qubit_gate[qubit]:]:
                    hw_other = qubit_hardware_map[gate.pair[1]]
                    if gate.pair[0] == qubit and hw_other != -1:
                        max_hardware_dist = max(max_hardware_dist, self.hw_distances.swap_distance[hw_qubit][hw_other])
            qubit += 1
        return max_hardware_dist


class LevelZeroQubitDistanceSwapEstimator(RemainingSwapsEstimator):

    def __init__(self, hw_distances : HardwareDistance) -> None:

        super().__init__(hw_distances)
        self.encoding = hw_distances.encoding


    def remain_swaps_estimate(self, next_qubit_gate: list[int], qubit_hardware_map: list[int]) -> int:

        max_hardware_dist = 0
        qubit = 0
        for hw_qubit in  qubit_hardware_map:
            if hw_qubit != -1 and next_qubit_gate[qubit] < self.encoding.nmb_qubit_gates[qubit]:
                gate = self.encoding.qubit_gates[qubit][next_qubit_gate[qubit]]
                hw_other = qubit_hardware_map[gate.pair[1]]
                if gate.pair[0] == qubit and hw_other != -1 and self.encoding.gate_position_for_qubit[gate.id][1] == next_qubit_gate[gate.pair[1]]:
                    max_hardware_dist = max(max_hardware_dist, self.hw_distances.swap_distance[hw_qubit][hw_other])
            qubit += 1
        return max_hardware_dist
