from problem import Problem, ProblemGate


class GateEncoding:

    def __init__(self, id : int, pair : tuple[int, int], duration : int) -> None:
        self.id = id
        self.pair = pair
        self.duration = duration

    def __repr__(self):
        return f"g{self.id}-(q{self.pair[0]},q{self.pair[1]})-{self.duration}"


class ProblemEncoding:
    """An encoding of the input problem for the dynamic graph algorithm."""


    problem : Problem
    """The input problem."""

    logical_qubit_names : list[str]
    """The names of the logical qubits as given in the input problem."""

    hardware_qubit_names : list[str]
    """The names of the hardware qubits as given in the input problem."""

    nmb_logical_qubits : int
    """The number of logical qubits in the input problem."""

    nmb_hardware_qubits : int
    """The number of hardware qubits in the input problem."""

    hardware_edges : list[tuple[int, int]]
    """The edges in the hardware graph. Each edge is given by a tuple of the indices of the two hardware qubits in the edge."""

    gates : list[GateEncoding]
    """The gates in the order they are given in the input problem."""

    qubit_gates : list[list[GateEncoding]]
    """The gates for each logical qubit in the order they are given in the input problem."""

    gate_position_for_qubit : dict[int, tuple[int, int]]
    """For each gate, the position that gate has in the list of gates for each of the two qubits of the gate.
    Key is gate id, Value is position of gate in list of gates for first qubit, and position of gate in list of gates for seqcond qubit."""

    nmb_qubit_gates : list[int]
    """The number of qubit gates for each logical qubit."""

    def __init__(self, problem : Problem) -> None:

        self.problem = problem

        # Get qubit names, position is index used for qubit in algorithm
        self.logical_qubit_names = self.problem.logical_qubit_names()
        self.hardware_qubit_names = self.problem.hardware_qubit_names()
        self.nmb_logical_qubits = len(self.logical_qubit_names)
        self.nmb_hardware_qubits = len(self.hardware_qubit_names)

        # Build hardware edges
        self.hardware_edges = self._build_hardware_edges()

        # Create gate lists
        self.gates = self._build_gates()
        self.qubit_gates = list([] for _ in range(self.nmb_logical_qubits))
        self.gate_position_for_qubit = {}
        self._build_qubit_gates()
        self.nmb_qubit_gates = list(len(q_gates) for q_gates in self.qubit_gates)


    def _build_hardware_edges(self) -> list[tuple[int, int]]:

        name_to_q = {}
        for idx in range(self.nmb_hardware_qubits):
            name_to_q[self.hardware_qubit_names[idx]] = idx
        return [(name_to_q[n1], name_to_q[n2]) for n1, n2 in self.problem.hardware_edges]


    def _build_gates(self) -> list[GateEncoding]:

        name_to_q = {}
        for idx in range(self.nmb_logical_qubits):
            name_to_q[self.logical_qubit_names[idx]] = idx
        return [GateEncoding(pb_gate.id, (name_to_q[pb_gate.pair[0]], name_to_q[pb_gate.pair[1]]), pb_gate.duration) for pb_gate in self.problem.gates]


    def _build_qubit_gates(self) -> None:

        for gate in self.gates:
            self.gate_position_for_qubit[gate.id] = tuple(len(self.qubit_gates[qubit]) for qubit in gate.pair)
            for qubit in gate.pair:
                self.qubit_gates[qubit].append(gate)


    def build_gate_layers(self) -> list[int]:

        gate_layers = []
        for gate in self.gates:
            pos = self.gate_position_for_qubit[gate.id]
            gate_layers.append(1 + max(-1 if pos[idx] == 0 else gate_layers[self.qubit_gates[gate.pair[idx]][pos[idx] - 1].id] for idx in range(2)))
        return gate_layers
    # def build_qubit_gate_layers(self) -> list[list[int]]:

    #     qubit_gate_layers = [[0] * len(self.qubit_gates[idx]) for idx in range(self.nmb_logical_qubits)]
    #     for gate in self.gates:
    #         pos = self.gate_position_for_qubit[gate.id]
    #         layer = 1 + max(-1 if pos[idx] == 0 else qubit_gate_layers[gate.pair[idx][pos[idx] - 1].id] for idx in range(2))
    #         for idx in range(2):
    #             qubit_gate_layers[gate.pair[idx][pos[idx]]] = layer
    #     return qubit_gate_layers
