from hardware_distance import HardwareDistance

class DepthEstimator:
    """Base class for an admissible heuristic of the final depth needed to complete from a given node state in the dynamic algorithm."""

    def depth_estimate(self, hw_qubit_depth: list[int], next_qubit_gate: list[int], qubit_hardware_map: list[int], hardware_qubit_map: list[int]) -> int:
        pass

class RemainingSwapsEstimator:

    def __init__(self, hw_distances : HardwareDistance) -> None:
        self.hw_distances = hw_distances

    def remain_swaps_estimate(self, next_qubit_gate: list[int], qubit_hardware_map: list[int]) -> int:
        pass
