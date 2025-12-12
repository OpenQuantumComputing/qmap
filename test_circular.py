from test_tools import circular_hardware_graph, to_problem_input_json
from dynamic_graph import DynamicGraph, DynamicGraphParameters, DivingParameters, NodeGrouping
from problem import Problem
from problem_encoding import ProblemEncoding
from hardware_distance import HardwareDistance
from depth_estimators import GateDependencyDepths, IndexedPathSwapsDepthEstimator
from swap_estimators import AllGatesQubitDistanceSwapEstimator


if __name__=="__main__":

    hw_edges = circular_hardware_graph(6)
    gate_indices = [
        (0, 1), (2, 3), (4, 5),
        (0, 5), (1, 2), (3, 4),
        (0, 1), (2, 3), (4, 5),
        (0, 2)]

    problem_json = to_problem_input_json(gate_indices, hw_edges, 1, 3)
    print(f"problem = {problem_json}")
    problem = Problem(problem_json)

    encoding = ProblemEncoding(problem)
    hw_distances = HardwareDistance(encoding, True)
    gate_depths = GateDependencyDepths(encoding)
    depth_estimator = IndexedPathSwapsDepthEstimator(gate_depths, hw_distances)
    swap_estimator = AllGatesQubitDistanceSwapEstimator(hw_distances)
    dynamic_graph_parameters = DynamicGraphParameters(DivingParameters(-1, 0, 0), 0, 1, 0, 0, 0, NodeGrouping.NONE, False, 0.0)
    dynamic_graph = DynamicGraph(encoding, dynamic_graph_parameters, depth_estimator, swap_estimator)
    solution, _ = dynamic_graph.find_solution(None)
    solution_json = solution.to_json_dict()
    print(f"solution = {solution_json}")
