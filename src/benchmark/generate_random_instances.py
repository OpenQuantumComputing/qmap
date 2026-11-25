import json
from pathlib import Path

import hardware_graphs
import networkx as nx
import numpy as np
from common import INSTANCES_ROOT, add_qmap_to_import_path
from qiskit import QuantumCircuit, qpy, transpile
from qiskit.circuit.random import random_circuit
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_experiments.library import QuantumVolume

add_qmap_to_import_path()
import converter


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


def remove_one_qubit_gates(circuit: QuantumCircuit) -> QuantumCircuit:
    result = QuantumCircuit(circuit.num_qubits)
    for instruction in circuit.data:
        if len(instruction.qubits) > 1:
            result.append(
                instruction.operation,
                instruction.qubits,
                instruction.clbits,
            )
    return result


def _write_instance(
    circuit: QuantumCircuit,
    hardware_graph: nx.Graph,
    output_directory: Path,
    filename: str,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    representation = converter.GateRepresentation(circuit)
    problem = converter.build_problem_dict(
        representation,
        list(hardware_graph.edges),
        converter.swap_duration,
    )
    with (output_directory / f"{filename}.json").open("w", encoding="utf-8") as file:
        json.dump(problem, file, indent=2)
    with (output_directory / f"{filename}.qpy").open("wb") as file:
        qpy.dump(circuit, file)


def gen_instances_quantum_volume(
    hardware_graph: nx.Graph,
    hardware_graph_name: str,
    num_qubits: int,
    output_directory: Path,
    seed: int = 42,
    num_instances: int = 10,
) -> None:
    backend = ArtificialBackend(num_qubits)
    experiment = QuantumVolume(
        tuple(range(num_qubits)),
        seed=seed,
        trials=num_instances,
    )
    for index, circuit in enumerate(experiment.circuits()):
        transpiled = transpile(
            circuit,
            backend=backend,
            optimization_level=3,
            seed_transpiler=seed,
        )
        _write_instance(
            transpiled,
            hardware_graph,
            output_directory,
            f"problem_QV_{hardware_graph_name}_{index}",
        )


def gen_instances_random(
    hardware_graph: nx.Graph,
    hardware_graph_name: str,
    num_qubits: int,
    depth: int,
    output_directory: Path,
    seed: int = 42,
    num_instances: int = 10,
    progress_prefix: str | None = None,
) -> None:
    rng = np.random.default_rng(seed)
    backend = ArtificialBackend(num_qubits)
    for index in range(num_instances):
        circuit_seed = int(rng.integers(1_000_000_000))
        circuit = random_circuit(
            num_qubits=num_qubits,
            depth=depth,
            seed=circuit_seed,
            max_operands=2,
        )
        transpiled = transpile(
            circuit,
            backend=backend,
            optimization_level=0,
            seed_transpiler=circuit_seed,
        )
        _write_instance(
            transpiled,
            hardware_graph,
            output_directory,
            f"problem_random_{hardware_graph_name}_{depth}_{index}",
        )
        if progress_prefix is not None:
            print(f"{progress_prefix}: {index + 1}/{num_instances}", flush=True)


def gen_instances_Y(
    different_qubits=range(4, 7),
    show_progress: bool = False,
) -> None:
    for num_qubits in different_qubits:
        graph, graph_name = hardware_graphs.Y_coupling(num_qubits)
        for depth in (10, 15, 20):
            gen_instances_random(
                graph,
                graph_name,
                num_qubits,
                depth,
                INSTANCES_ROOT / "Y",
                num_instances=100,
                progress_prefix=(
                    f"Y-{num_qubits} depth {depth}" if show_progress else None
                ),
            )


def gen_instances_grid() -> None:
    for rows, columns in ((2, 2), (2, 3)):
        graph, graph_name = hardware_graphs.grid_coupling(rows, columns)
        for depth in (10, 15, 20):
            gen_instances_random(
                graph,
                graph_name,
                rows * columns,
                depth,
                INSTANCES_ROOT / "grid",
                num_instances=100,
            )


def gen_instances_linear(
    different_qubits=range(4, 7),
    show_progress: bool = False,
) -> None:
    for num_qubits in different_qubits:
        graph, graph_name = hardware_graphs.linear_coupling(num_qubits)
        for depth in (10, 15, 20):
            gen_instances_random(
                graph,
                graph_name,
                num_qubits,
                depth,
                INSTANCES_ROOT / "linear",
                num_instances=100,
                progress_prefix=(
                    f"Linear-{num_qubits} depth {depth}" if show_progress else None
                ),
            )


if __name__ == "__main__":
    gen_instances_grid()
