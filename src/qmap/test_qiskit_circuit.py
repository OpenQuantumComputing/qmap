import converter
from qiskit import QuantumCircuit

import run_dynamic_graph as dg_solver
from dynamic_graph import NodeGrouping,  DynamicGraphParameters, DivingParameters

import json

if __name__ == '__main__':

    # Create quantum circuit
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.h(0)
    qc.cx(1, 2)
    qc.cx(1, 3)
    qc.h(1)
    qc.h(2)
    qc.x(3)
    qc.h(3)
    qc.x(3)
    print(qc.draw())

    # convert from circuit to representation
    rep = converter.GateRepresentation(qc)

    # build problem
    pd = converter.build_problem_dict(rep, [['p1', 'p2'], ['p2', 'p3'], ['p3', 'p4']], 3)
    with open('problem.json', 'w') as fh:
        json.dump(pd, fh)

    # run_dynamic_graph.py : problem.json -> solution.json

    single_file_in="problem.json"
    single_file_out="solution.json"

    swaps_objective = 0
    depth_objective = 1
    diving_frequency = 2000
    nmb_top_gate_dives = 20
    max_gates_top_dive = 6
    statistics_grouping : NodeGrouping = NodeGrouping.NONE
    unassigned_objective = 0
    beam_width = 0
    remain_gates_weight = 0
    layering_active = False
    timeout_seconds=150
    parameters = DynamicGraphParameters(DivingParameters(diving_frequency, nmb_top_gate_dives, max_gates_top_dive), swaps_objective, depth_objective, unassigned_objective, beam_width, remain_gates_weight, statistics_grouping, layering_active, timeout_seconds)

    dg_solver.solve_single_problem(single_file_in, single_file_out, parameters)


    # load solution
    with open('solution.json', 'r') as fh:
        solution = json.load(fh)

    # convert from representation + solution to circuit
    qc2 = rep.to_circuit(solution['dynamic_graph_solution'])
    print(qc2.draw())  # note that isolated single gates are missing