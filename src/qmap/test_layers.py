import unittest
from test_tools import linear_hardware_graph, to_problem_input_json, no_dive_parameters
from dynamic_graph import DynamicGraph, DynamicGraphParameters, DivingParameters, NodeGrouping
from problem import Problem
from problem_encoding import ProblemEncoding
from hardware_distance import HardwareDistance
from depth_estimators import GateDependencyDepths, IndexedPathSwapsDepthEstimator
from swap_estimators import AllGatesQubitDistanceSwapEstimator
from solution import Solution


def _get_test_solution(encoding : ProblemEncoding, layering_active : bool) -> Solution:

    hw_distances = HardwareDistance(encoding, True)
    gate_depths = GateDependencyDepths(encoding)
    depth_estimator = IndexedPathSwapsDepthEstimator(gate_depths, hw_distances)
    swap_estimator = AllGatesQubitDistanceSwapEstimator(hw_distances)
    dynamic_graph_parameters = DynamicGraphParameters(no_dive_parameters(), 0, 1, 0, 0, 0, NodeGrouping.NONE, layering_active, 0.0)

    dynamic_graph = DynamicGraph(encoding, dynamic_graph_parameters, depth_estimator, swap_estimator)
    solution, _ = dynamic_graph.find_solution(None)

    return solution

class DynamicGraphLayeringTests(unittest.TestCase):

    def test_layering_gives_different_solution(self):

        hw_edges = linear_hardware_graph(8)
        gate_indices = [
            (0, 1), (2, 3), (4, 5), (6,7),
            (1, 2), (3, 4), (5, 6),
            (0, 1), (2, 3), (4, 5), (6,7),
            (1, 2), (3, 4), (5, 6),
            (2, 5), (3, 4),
            (3, 4)]

        problem_json = to_problem_input_json(gate_indices, hw_edges, 1, 3)
        # print(f"problem = {problem_json}")
        problem = Problem(problem_json)
        encoding = ProblemEncoding(problem)

        non_layer_solution = _get_test_solution(encoding, False)
        layer_solution = _get_test_solution(encoding, True)
        # print(f"Non-layer solution = {non_layer_solution.to_json_dict()}")
        # print(f"Layer solution = {layer_solution.to_json_dict()}")

        self.assertEqual(10, non_layer_solution.depth())
        self.assertEqual(12, layer_solution.depth())

if __name__=="__main__":
    unittest.main()

