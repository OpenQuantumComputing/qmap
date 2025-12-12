import sys
import json
import os
import glob
import getopt
import time
from problem import Problem
from problem_encoding import ProblemEncoding
from dynamic_graph import DynamicGraph, DynamicGraphParameters, DivingParameters, NodeGrouping, TerminationCause
from solution import Solution, SolutionJob
from swap_estimators import AllGatesQubitDistanceSwapEstimator
from hardware_distance import HardwareDistance
from depth_estimators import GateDependencyDepths, LeastSwapsDepthEstimator, PathSwapsDepthEstimator, IndexedPathSwapsDepthEstimator

def to_solution(problem : Problem, optimality_ensured : bool, solution_json) -> Solution:

    hardw_to_logical = solution_json["bit_assignment"]
    jobs : list[SolutionJob] = []
    for operation in solution_json["operations"]:
        start = operation["time"]
        gate_idx : int = operation["gate"]
        gate = None if gate_idx == None else problem.gates[gate_idx]
        duration = problem.swap_time if gate == None else gate.duration
        edge = operation["edge"]
        jobs.append(SolutionJob(start, duration, gate, (edge[0], edge[1])))
    return Solution(problem, optimality_ensured, hardw_to_logical, jobs)


def solve_single_problem(file_path_in : str, file_path_out : str, parameters : DynamicGraphParameters) -> tuple[int, int, int]:

    with open(file_path_in, "r") as input_file:
        data = json.load(input_file)
    print(f"Input read from {file_path_in}")
    problem_json = data["problem"]
    problem = Problem(problem_json)

    encoding = ProblemEncoding(problem)
    hw_distances = HardwareDistance(encoding, True)
    gate_depths = GateDependencyDepths(encoding)
    depth_estimator = IndexedPathSwapsDepthEstimator(gate_depths, hw_distances)
    swap_estimator = AllGatesQubitDistanceSwapEstimator(hw_distances)
    # swap_estimator = LevelZeroQubitDistanceSwapEstimator(hw_distances)
    dynamic_graph = DynamicGraph(encoding, parameters, depth_estimator, swap_estimator)
    print(f"Solving for {file_path_in}")
    solution, cause = dynamic_graph.find_solution(None)
    # solution = dynamic_graph.find_solution(None if non_layer_solution == None else non_layer_solution.hardware_to_logical)

    if cause == TerminationCause.EXHAUSTED:
        if solution == None:
            print("Algorithm terminated without finding any solution")
        else:
            raise Exception("Unexpected result: Algorithm terminated as EXHAUSTED, but also returned a solution.")
    elif cause == TerminationCause.TIMEOUT:
        if solution == None:
            print("Algorithm timed out, no solution found.")
        else:
            print("Algorithm timed out, using best solution found before timeout.")

    if solution != None:
        validation_error = solution.validate(problem.swap_time)
        solution_depth = solution.depth()
        solution_swaps = solution.swaps()

        if validation_error == None:
            print(f"Solution depth = {solution_depth}")
            print(f"Solution swaps = {solution_swaps}")
            if file_path_out != None:
                solution_json = solution.to_json_dict()
                dynamic_graph_solution = {"solution": solution_json}
                data["dynamic_graph_solution"] = dynamic_graph_solution
                json_object = json.dumps(data)
                # json_object = json.dumps(data, indent=4)
                with open(file_path_out, "w") as outfile:
                    outfile.write(json_object)
                print(f"Solution added to input data and stored to {file_path_out}")
        else:
            solution_depth = None
            solution_swaps = None
            print(f"Dynamic graph solution validation failed: {validation_error}")
    else:
        solution_depth = None
        solution_swaps = None

    return (len(problem.gates), solution_depth, solution_swaps)


def solve_multi_problems(input_dir : str, output_dir : str, reg_exp : str, parameters : DynamicGraphParameters) -> None:

    search_path = os.path.join(input_dir, reg_exp)
    files = glob.glob(search_path)
    bad_runs = 0
    log_lines = []
    for file_path_in in files:
        file_path_in = file_path_in.replace("\\", "/")
        if output_dir == None:
            file_path_out = None
        else:
            file_path_out = file_path_in.replace(input_dir, output_dir)
        t0 = time.time()
        nmb_gates, sol_depth, sol_swaps = solve_single_problem(file_path_in, file_path_out, parameters)
        t1 = time.time()
        solve_time = t1 - t0
        if sol_depth == None:
            bad_runs += 1
        # loc_file_name = file_path_in.replace(input_dir, "")
        # log_lines.append(f"{loc_file_name}\t{nmb_gates}\t{'?' if sol_depth == None else sol_depth}\t{'?' if sol_swaps == None else sol_swaps}\t{solve_time}")
        log_lines.append(f"{file_path_in} gates={nmb_gates} depth={'?' if sol_depth == None else sol_depth} swaps={'?' if sol_swaps == None else sol_swaps} time={solve_time}")

    print()
    if bad_runs > 0:
        print(f"For {bad_runs} problems, either no solution was found or the optimal solution was missing in the input")
    print()
    # print("File\tGates\tDepth\tSwaps\tTime")
    for ll in log_lines:
        print(ll)


def parse_int(value: str, default: int) -> int:

    try:
        result = int(value)
    except ValueError:
        result = default
    return result


def parse_float(value: str, default: float) -> float:

    try:
        result = float(value)
    except ValueError:
        result = default
    return result


def parse_statistics_grouping(value: str, default: NodeGrouping) -> NodeGrouping:

    value_up = value.upper()

    if value_up in ("T", "TRUE", "1", "Y", "YES", "GATES", "REMAINING_GATES"):
        return NodeGrouping.REMAINING_GATES
    elif value_up in ("F", "FALSE", "0", "N", "NO", "NONE"):
        return NodeGrouping.NONE
    elif value_up in ("LEVEL"):
        return NodeGrouping.LEVEL
    elif value_up in ("SWAPS", "ADDED_SWAPS"):
        return NodeGrouping.ADDED_SWAPS
    else:
        return default


def parse_bool(value: str, default: bool) -> bool:

    value_up = value.upper()

    if value_up in ("T", "TRUE", "1", "Y", "YES"):
        return True
    elif value_up in ("F", "FALSE", "0", "N", "NO"):
        return False
    else:
        return default


if __name__=="__main__":

    """
    Parameter options when calling 'run_dynamic_graph'

    Parameters for file input:
    -i <input json file>
        Sets the input file, necessary. Expected to be json file with format as in ../timeindexed/experiments/layers_nonoptimal/
    -o <output json file>
        Path for output file, optional. Will be same as input file with dynamic_graph solution added. If not set, no solution is written to file.

    Parameters for objective, optional:
    -d <weight>
        Sets the weight of the depth in the objective, must be integer. Default is 1.
    -s <weight>
        Sets the weight of the number of swaps in the objective, must be integer. Default is 0 (disabled).
    -u <weight>
        Sets the weight of the number of unassigned qubits in the objective, must be integer. Default is 0 (disabled).

    Parameters for diving (all optional):
    See variable documentation in class DivingStrategy in dynamic_graph.py for details.
    To disable diving, use -f -1 -g 0
    -D <boolean value>
        Whether diving is on or off
        If True, use diving values as set by -f, -g and -m
        If False, disables diving by overriding -f and -g to -f -1 -g 0
        Default is True
    -f <value>
        Sets the diving_frequency variable in DivingStrategy. Default is 2000.
    -g <value>
        Sets the nmb_top_gate_dives variable in DivingStrategy. Default is 20.
    -m <value>
        Sets the max_gates_top_dive variable in DivingStrategy. Default is 6.

    Paramters for heuristic type (optional)
    -b <value>
        Sets the width used in the beam search.
        If positive, the set of produced nodes at each level in the search tree will be pruned to contain only the N nodes with best heuristic value, where N is the beam width.
        If 0, beam search is disabled.
        Default is 0.

    Paramters for layering (optional)
    -L <boolean value>
        Whether layer division is activated or not
        If True, layer division is activated. The gates are divided into layers, a node in the search tree can not plan a gate G if there are unplanned gates in lower layers than the layer of G.
        A gate is in the lowest possible layer such that all of its preceding gates are in lower layers. Layer 0 are the gates with no preceding gates.
        If False, layer division is inactive. Any unplanned gate can be planned as long as its preceding gates are planned, even if there are other unplanned gates in lower layers.
        Default is False.

    Paramters for heuristic type (optional)
    -a <boolean value>
        This option is deprecated, use '-l 0' for '-a True' and '-l 1' for '-a False'
        If value is True, use admissible heuristic in A* search, the solution returned should be optimal, but may take time.
        If value is False, non-admissible heuristic is used (number of remaining gates is added to admissible heuristic). The returned solution can often be optimal, but not always. Often faster than admissible heuristic.
        Default is True.
    -l <value>
        Sets the weight on the number of remaining gates to be added the heuristic value. If 0, the heuristic will be admissible and the solution returned should be optimal, but may take time.
        If positive, the heuristic will be non-admissible. The algorithm will typically terminate faster but the returned solution may not be optimal. Higher values gives faster termination and higher risk of deviation from optimal.
        Default is 0 (admissible heuristic)

    Paramters for timeout (optional)
    -t <value>
        Number of seconds the algorithm can run before terminating with timeout if the dynamic graph tree search has not reached any end node yet.
        If the algorithm terminates due to timeout, the best solution found by the diving mechanism is returned. If no such diving solution exists, no solution is returned.
        If the value is 0.0, the algorithm will not terminate due to timeout.
        Default is 0.0, i.e. no timeout limit is applied.

    Parameters normally not relevant, used for running a batch of several problems, or for making statistics on heuristic quality.
    -I, -O, -r, -S
    """

    opt_arguments = sys.argv[1:]

    single_file_in = None
    single_file_out = None
    input_dir = None
    output_dir = None
    reg_exp = None
    swaps_objective = 0
    depth_objective = 1
    unassigned_objective = 0
    diving_enabled = True
    diving_frequency = 2000
    nmb_top_gate_dives = 20
    max_gates_top_dive = 6
    beam_width = 0
    statistics_type : str = None
    statistics_grouping : NodeGrouping = NodeGrouping.NONE
    admissible_heuristic = True
    remain_gates_weight = 0
    layering_active = False
    timeout_seconds = 0.0

    options = "a:b:d:D:f:g:I:i:L:l:m:O:o:r:s:S:t:u:"
    long_options = ["admissible=", "beam_width=", "depth_objective=", "dive=", "frequency_diving=", "gates_dives=", "input_directory=", "input_file=", "layering_active=", "remain_gates_weight=", "max_gates_dives=", "output_directory=", "output_file=", "reg_exp=", "swaps_objective=", "statistics=", "timeout=", "unassigned_objective="]

    try:
        arguments, values = getopt.getopt(opt_arguments, options, long_options)

        for argument, value in arguments:

            if argument in ("-a", "--admissible"):
                remain_gates_weight = 0 if parse_bool(value, True) else 1
            elif argument in ("-b", "--beam_width"):
                beam_width = parse_int(value, 1)
            elif argument in ("-d", "--depth_objective"):
                depth_objective = parse_int(value, 1)
            elif argument in ("-D", "--dive"):
                diving_enabled = parse_bool(value, True)
            elif argument in ("-f", "--frequency_diving"):
                diving_frequency = parse_int(value, 2000)
            elif argument in ("-g", "--gates_dives"):
                nmb_top_gate_dives = parse_int(value, 20)
            elif argument in ("-I", "--input_directory"):
                input_dir = value.replace("\\","/")
            elif argument in ("-i", "--input_file"):
                single_file_in = value.replace("\\","/")
            elif argument in ("-L", "--layering_active"):
                layering_active = parse_bool(value, True)
            elif argument in ("-l", "--remain_gates_weight"):
                remain_gates_weight = parse_int(value, 0)
            elif argument in ("-m", "--max_gates_dives"):
                max_gates_top_dive = parse_int(value, 6)
            elif argument in ("-O", "--output_directory"):
                output_dir = value.replace("\\","/")
            elif argument in ("-o", "--output_file"):
                single_file_out = value.replace("\\","/")
            elif argument in ("-r", "--reg_exp"):
                reg_exp = value.replace("\\","/")
            elif argument in ("-s", "--swaps_objective"):
                swaps_objective = parse_int(value, 0)
            elif argument in ("-S", "--statistics"):
                statistics_grouping = parse_statistics_grouping(value, NodeGrouping.REMAINING_GATES)
            elif argument in ("-t", "--timeout"):
                timeout_seconds = parse_float(value, 0.0)
            elif argument in ("-u", "--unassigned_objective"):
                unassigned_objective = parse_int(value, 1)

        if not diving_enabled:
            #-f -1 -g 0
            diving_frequency = -1
            nmb_top_gate_dives = 0

        parameters = DynamicGraphParameters(DivingParameters(diving_frequency, nmb_top_gate_dives, max_gates_top_dive), swaps_objective, depth_objective, unassigned_objective, beam_width, remain_gates_weight, statistics_grouping, layering_active, timeout_seconds)
        if single_file_in != None:
            solve_single_problem(single_file_in, single_file_out, parameters)
        if input_dir != None:
            if reg_exp == None:
                print("Can not run for all files in directory if not reg_exp is set")
            elif statistics_grouping != NodeGrouping.NONE:
                print("Can not run for all files in directory with statistics turned on")
            else:
                solve_multi_problems(input_dir, output_dir, reg_exp, parameters)

    except getopt.error as err:
        print(str(err))
