import json
from pathlib import Path

import hardware_graphs
import networkx as nx
import numpy as np
from common import BENCHMARK_ROOT, add_qmap_to_import_path
from qiskit import transpile
from qiskit.circuit.random import random_circuit
from qiskit.providers.fake_provider import GenericBackendV2

add_qmap_to_import_path()
import converter
import run_dynamic_graph as dg_solver
from dynamic_graph import DivingParameters, DynamicGraphParameters, NodeGrouping


class ArtificialBackend(GenericBackendV2):
    def __init__(self, num_qubits: int) -> None:
        graph = nx.complete_graph(num_qubits)
        coupling_map = list(graph.edges) + [
            (right, left) for left, right in graph.edges
        ]
        super().__init__(
            num_qubits=num_qubits,
            coupling_map=coupling_map,
            calibrate_instructions=True,
            seed=42,
        )


def _solver_parameters(
    layering: bool,
    objective: str,
) -> DynamicGraphParameters:
    if objective not in {"swaps", "depth"}:
        raise ValueError("objective must be either 'swaps' or 'depth'")
    return DynamicGraphParameters(
        DivingParameters(2000, 20, 6),
        int(objective == "swaps"),
        int(objective == "depth"),
        0,
        0,
        0,
        NodeGrouping.NONE,
        layering,
        500,
    )


def solve_example_instance(
    circuit,
    num_qubits: int,
    hardware_graph: nx.Graph,
    hardware_graph_name: str,
    depth: int,
    layering: bool = False,
    objective: str = "swaps",
    output_root: Path = BENCHMARK_ROOT / "solutions" / "ExampleInstance",
) -> None:
    parameters = _solver_parameters(layering, objective)
    instance_root = BENCHMARK_ROOT / "instances" / "ExampleInstance"
    instance_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    settings = parameters.__dict__.copy()
    settings["diving_parameters"] = settings["diving_parameters"].__dict__
    settings["statistics_grouping"] = 0
    with (output_root / "settings.json").open("w", encoding="utf-8") as file:
        json.dump(settings, file, indent=4)

    backend = ArtificialBackend(num_qubits)
    transpiled = transpile(
        circuit,
        backend=backend,
        optimization_level=3,
        seed_transpiler=42,
    )
    representation = converter.GateRepresentation(transpiled)
    problem = converter.build_problem_dict(
        representation,
        list(hardware_graph.edges),
        converter.swap_duration,
    )
    filename = f"problem_random_{hardware_graph_name}_{depth}.json"
    input_path = instance_root / filename
    output_path = output_root / filename
    with input_path.open("w", encoding="utf-8") as file:
        json.dump(problem, file, indent=2)

    dg_solver.solve_single_problem(str(input_path), str(output_path), parameters)


if __name__ == "__main__":
    seed = 42
    rng = np.random.default_rng(seed)
    circuit_seed = int(rng.integers(1_000_000_000))
    num_qubits = 4
    depth = 10
    circuit = random_circuit(
        num_qubits=num_qubits,
        depth=depth,
        seed=circuit_seed,
        max_operands=2,
    )
    graph, graph_name = hardware_graphs.linear_coupling(num_qubits)
    solve_example_instance(
        circuit,
        num_qubits,
        graph,
        graph_name,
        depth,
    )
