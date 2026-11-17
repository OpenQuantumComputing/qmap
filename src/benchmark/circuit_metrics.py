import json
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.exceptions import QiskitError
from qiskit.providers.fake_provider import Fake20QV1, GenericBackendV2
from qiskit.scheduler.config import ScheduleConfig
from qiskit.scheduler.schedule_circuit import schedule_circuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

BENCHMARK_ROOT = Path(__file__).resolve().parent


def _load_pickle(filename):
    path = BENCHMARK_ROOT / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing benchmark support file: {path}. "
            "Generate or provide the benchmark noise model artifacts first."
        )
    with path.open("rb") as file:
        return pickle.load(file)


def remove_one_qubit_gates(circuit):

    new_qc = QuantumCircuit(circuit.num_qubits)
    # Iterate through instructions and add only multi-qubit gates to the new circuit
    for instr, qargs, cargs in circuit.data:
        if len(qargs) > 1:
            new_qc.append(instr, qargs, cargs)
    return new_qc


def remove_measure_gates(circuit):
    new_qc = QuantumCircuit(circuit.num_qubits)
    # Iterate through instructions and add only multi-qubit gates to the new circuit
    for instr, qargs, cargs in circuit.data:
        if not instr.name == "measure":
            new_qc.append(instr, qargs, cargs)
    return new_qc


def METRIC_1_pulse_time(
    circuit, backend, method="asap", remove_one_qubit_gates_bool=False, **tp_kwargs
):
    # print(circuit.draw())
    transpiled_circuit = transpile(circuit, backend=backend, **tp_kwargs)
    # print(transpiled_circuit.draw())
    if remove_one_qubit_gates_bool:
        transpiled_circuit = remove_one_qubit_gates(transpiled_circuit)
    # print(transpiled_circuit.draw())
    schedule_config = ScheduleConfig(
        backend.instruction_schedule_map, backend.meas_map, backend.dt
    )
    try:
        schedule = schedule_circuit(
            transpiled_circuit, schedule_config, method, backend
        )
    except QiskitError:
        return None
    return schedule.duration


def compare_results(probabilities_A, probabilities_B):
    keys = set(list(probabilities_A.keys()) + list(probabilities_B.keys()))
    delta = 0
    for key in keys:
        p_A = probabilities_A.get(key, 0.0)
        p_B = probabilities_B.get(key, 0.0)
        dp = abs(p_A - p_B)  # use simple abs dist metric for proba distance
        delta += dp
    return delta


def METRIC_2_noise_effect(circuit, noise_model, seed_simulator, gates, **tp_kwargs):
    circuit_copy_ideal = circuit.copy()
    circuit_copy_ideal = remove_measure_gates(circuit_copy_ideal)
    circuit_copy_ideal.save_state()

    circuit_copy_noise = circuit.copy()
    if circuit_copy_noise.num_clbits == 0:
        circuit_copy_noise.measure_all()

    rng = np.random.default_rng(seed_simulator)

    noisy_simulator = AerSimulator(
        method="density_matrix", noise_model=noise_model, basis_gates=gates, shots=1024
    )
    ideal_simulator = AerSimulator(
        method="statevector",
        seed_simulator=seed_simulator,
        noise_model=None,
        basis_gates=gates,
    )

    transpiled_circuit_ideal = transpile(
        circuit_copy_ideal, backend=noisy_simulator, **tp_kwargs
    )
    ideal_result = ideal_simulator.run(transpiled_circuit_ideal).result()
    ideal_probabilities = ideal_result.get_statevector().probabilities_dict()

    transpiled_circuit_noise = transpile(
        circuit_copy_noise, backend=noisy_simulator, **tp_kwargs
    )

    results_probabilities = []
    # results_fidelities=[]
    num_simulations = 1

    # It would be good to run this in parallel
    for i in range(num_simulations):
        current_seed = rng.integers(1e9)
        noisy_result = noisy_simulator.run(
            transpiled_circuit_noise, seed_simulator=current_seed
        ).result()

        counts = noisy_result.get_counts()

        # Calculate the probabilities
        total_shots = sum(counts.values())
        probabilities = {state: count / total_shots for state, count in counts.items()}
        results_probabilities.append(probabilities)

    all_keys = set()
    for elem in results_probabilities:
        all_keys.update(elem.keys())
    noisy_probabilities = {
        key: sum(prob.get(key, 0) for prob in results_probabilities) / num_simulations
        for key in all_keys
    }

    eps = 1e-9
    values_ideal = list(ideal_probabilities.values())
    for i in range(2**circuit.num_qubits - len(values_ideal)):
        values_ideal.append(0)
    median_prob = np.median(values_ideal)
    heavy_outputs = [
        key for key, prob in ideal_probabilities.items() if prob > median_prob + eps
    ]
    heavy_output_prob = sum(noisy_probabilities.get(key, 0) for key in heavy_outputs)

    return compare_results(noisy_probabilities, ideal_probabilities), heavy_output_prob


def METRIC_3_number_of_gates(circuit, gate_name="swap"):
    ops = circuit.count_ops()
    cnot_count = ops.get(gate_name, 0)
    return cnot_count


# optimized qubit map
def get_qubit(circuit, idx):
    for qreg in circuit.qregs:
        for qubit in qreg:
            found_idx, reg = circuit.find_bit(qubit)
            if idx == found_idx:
                return qubit
    raise ValueError


class ArtificialBackend(GenericBackendV2):
    def __init__(self, num_qubits, coupling_map):
        super().__init__(
            num_qubits=num_qubits,
            coupling_map=coupling_map,
            calibrate_instructions=True,
            seed=42,
        )


def evaluate_metrics(
    circuit,
    coupling_map,
    seed_simulator,
    seed_transpiler,
    remove_one_qubit_gates_bool=False,
):
    num_qubits = circuit.num_qubits
    map_logical_to_physical = {
        get_qubit(circuit, idx): idx for idx in range(num_qubits)
    }

    noise_model = _load_pickle("noise_model.pkl")

    tp_kwargs = dict(
        initial_layout=map_logical_to_physical,
        optimization_level=0,
        seed_transpiler=seed_transpiler,
    )

    # G=nx.Graph()
    # G.add_edges_from(coupling_map)
    G = nx.complete_graph(
        num_qubits
    )  # That's a bug fix. The edges aren't correctly read in the converter

    mapping = {node: i for i, node in enumerate(G.nodes())}
    G_integer = nx.relabel_nodes(G, mapping)

    edge_set = set(G_integer.edges())
    # Iterate over the original list and add the reverse tuples
    for a, b in G_integer.edges():
        edge_set.add((b, a))
    # Convert back to a list if needed
    edges = list(edge_set)

    backend = ArtificialBackend(num_qubits, edges)

    gates = _load_pickle("basis_gates.pkl")

    duration = METRIC_1_pulse_time(
        circuit,
        backend,
        remove_one_qubit_gates_bool=remove_one_qubit_gates_bool,
        **tp_kwargs,
    )

    noise_delta, heavy_output_prob = METRIC_2_noise_effect(
        circuit, noise_model, seed_simulator, gates, **tp_kwargs
    )  # note: use noise model from hardware backend

    gate_count = METRIC_3_number_of_gates(
        circuit
    )  # note: swap is converted to 3 cx during transpilation

    depth = circuit.depth()
    return dict(
        duration=duration,
        noise_delta=noise_delta,
        gate_count=gate_count,
        heavy_output_prob=heavy_output_prob,
        depth=depth,
    )


# To generate a new noise_model:


def generate_old_noise_model():
    raise NotImplementedError(
        "Use generate_new_noise_model to create benchmark noise artifacts."
    )


def generate_new_noise_model():
    hardware_backend = Fake20QV1()
    noise_model = NoiseModel.from_backend(hardware_backend)

    with (BENCHMARK_ROOT / "basis_gates.pkl").open("wb") as f:
        pickle.dump(hardware_backend.configuration().basis_gates, f)

    with (BENCHMARK_ROOT / "noise_model.json").open("w", encoding="utf-8") as f:
        json.dump(noise_model.to_dict(serializable=True), fp=f)
    with (BENCHMARK_ROOT / "noise_model.pkl").open("wb") as f:
        pickle.dump(noise_model, f)
