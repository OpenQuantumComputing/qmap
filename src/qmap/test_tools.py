from dynamic_graph import DivingParameters

def circular_hardware_graph(size : int) -> list[tuple[str, str]]:
    hw_qubits = [f"h{idx}" for idx in range(size)]
    return [[hw_qubits[idx], hw_qubits[0 if idx == size  - 1 else idx + 1]] for idx in range(size) ]

def linear_hardware_graph(size : int) -> list[tuple[str, str]]:
    hw_qubits = [f"h{idx}" for idx in range(size)]
    return [[hw_qubits[idx], hw_qubits[idx + 1]] for idx in range(size - 1) ]

def to_problem_input_json(gate_qubit_indices : list[tuple[int, int]], hardware_edges : list[tuple[str, str]], common_gate_duration : int, swap_time : int) -> dict:
    gates = [{"line1": f"q{q1}", "line2": f"q{q2}", "duration": common_gate_duration} for (q1, q2) in gate_qubit_indices]
    return {"gates": gates, "topology": hardware_edges, "swap_time": swap_time}

def no_dive_parameters() -> None:
    return DivingParameters(-1, 0, 0)