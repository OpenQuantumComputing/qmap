from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.converters import circuit_to_dag
from qiskit.converters import dag_to_circuit
from qiskit.transpiler.passes import Unroll3qOrMore


single_gate_duration=1
two_gate_duration=4
swap_duration=15


class Instruction():
    def __init__(self, instruction, circuit):
        self.clbit_indices = [circuit.find_bit(clbit).index for clbit in instruction.clbits]
        self.qubit_indices = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
        self.instruction = instruction


class Gate(Instruction):
    def __init__(self, instruction, circuit):
        super().__init__(instruction, circuit)


class GateCollection():
    def __init__(self, instructions, circuit):
        self.gates = [Gate(instruction, circuit) for instruction in instructions]


class OneQubitGate(Gate):
    def __init__(self, instruction, circuit, qubit_index):
        super().__init__(instruction, circuit)
        self.qubit_index = qubit_index
        self.duration = 1


class TwoQubitGate(GateCollection):
    def __init__(self, instructions, circuit, qubit_index1, qubit_index2,num_single_gates_index1=0,num_single_gates_index2=0):
        super().__init__(instructions, circuit)
        self.duration = two_gate_duration
        self.qubit_index1 = qubit_index1
        self.qubit_index2 = qubit_index2
        self.num_single_gates_index1=num_single_gates_index1
        self.num_single_gates_index2=num_single_gates_index2

def complete_dict(d, n):
    # Add missing keys with arbitrary value (-1 in this case)
    for i in range(n + 1):
        if i not in d:
            d[i] = -1

    # Find missing values
    all_values = set(range(n + 1))
    cur_values = set(d.values())
    missing_values = list(all_values - cur_values)
    
    # Replace arbitrary values with missing ones
    for key, value in d.items():
        if value == -1 and missing_values:
            d[key] = missing_values.pop(0)
    
    return d


class GateRepresentation():
    def __init__(self, circuit):
        self.two_qubit_gates = []
        self.isolated_one_qubit_gates = []  # append at the end of the circuit
        unrolled_circuit = self._unroll_circuit(circuit)
        self._from_unrolled_circuit(unrolled_circuit)

    def _unroll_circuit(self, circuit):
        dag = circuit_to_dag(circuit)
        unrolled_dag = Unroll3qOrMore().run(dag)
        unrolled_circuit = dag_to_circuit(unrolled_dag)
        return unrolled_circuit

    def _from_unrolled_circuit(self, unrolled_circuit):
        self._num_qubits = unrolled_circuit.num_qubits
        self._num_clbits = unrolled_circuit.num_clbits
        instruction_buffer = {idx: [] for idx in range(self._num_qubits)}
        for instruction in unrolled_circuit.data:
            if instruction.operation.num_qubits == 1:
                qubit_index = unrolled_circuit.find_bit(instruction.qubits[0]).index
                instruction_buffer[qubit_index].append(instruction)
            else:
                qubit_index1 = unrolled_circuit.find_bit(instruction.qubits[0]).index
                qubit_index2 = unrolled_circuit.find_bit(instruction.qubits[1]).index
                two_qubit_gate_instructions = instruction_buffer[qubit_index1].copy() + instruction_buffer[
                    qubit_index2].copy()
                num_single_gates_index1=len(instruction_buffer[qubit_index1])
                num_single_gates_index2=len(instruction_buffer[qubit_index2])
                instruction_buffer[qubit_index1].clear()
                instruction_buffer[qubit_index2].clear()
                two_qubit_gate_instructions.append(instruction)
                self.two_qubit_gates.append(
                    TwoQubitGate(two_qubit_gate_instructions, unrolled_circuit, qubit_index1, qubit_index2,num_single_gates_index1,num_single_gates_index2))
        # try to move single qubit gates to the left
        for idx, instructions in instruction_buffer.items():
            if len(instructions) > 0:
                for two_qubit_gate in self.two_qubit_gates[::-1]:
                    if two_qubit_gate.qubit_index1 == idx or two_qubit_gate.qubit_index2 == idx:
                        for instruction in instructions:
                            two_qubit_gate.gates.append(Gate(instruction, unrolled_circuit))
                        instructions.clear()
                        break
        # placeholder solution: store but ignore remaining isolated single qubit gates
        self.isolated_one_qubit_gates = [OneQubitGate(instruction, unrolled_circuit, qubit_index) for qubit_index in
                                         instruction_buffer.keys() for instruction in instruction_buffer[qubit_index]]

    def to_gates_json(self):
        gates = []
        for two_qubit_gate in self.two_qubit_gates:
            duration=two_qubit_gate.duration+single_gate_duration*max(two_qubit_gate.num_single_gates_index1,two_qubit_gate.num_single_gates_index2)
            gates.append(
                {'line1': f'l{two_qubit_gate.qubit_index1 + 1}', 'line2': f'l{two_qubit_gate.qubit_index2 + 1}',
                 'duration': duration})
        return gates

    def to_circuit(self, solution):
        solution = solution['solution']
        circuit = QuantumCircuit(self._num_qubits, self._num_clbits)
        bit_assignment_lp = {int(l[1:]) - 1: int(p[1:]) - 1 for p, l in solution['bit_assignment'].items()}
        bit_assignment_lp = complete_dict(bit_assignment_lp,self._num_qubits)
        for qubit_index in range(self._num_qubits):
            if qubit_index not in bit_assignment_lp:
                bit_assignment_lp[qubit_index] = qubit_index
        operations = sorted([d for d in solution['operations']], key=lambda d: d['time'])
        for operation in operations:
            gate_index = operation['gate']
            p1_index = int(operation['edge'][0][1:])
            p2_index = int(operation['edge'][1][1:])
            gate_qargs = [p1_index - 1, p2_index - 1]
            if type(gate_index) is int:
                qarg_map = {self.two_qubit_gates[gate_index].qubit_index1: gate_qargs[0],
                            self.two_qubit_gates[gate_index].qubit_index2: gate_qargs[1]}
                for gate in self.two_qubit_gates[gate_index].gates:
                    try:
                        qargs = [qarg_map[qubit_index] for qubit_index in gate.qubit_indices]
                        # print(qargs)
                        cargs = gate.clbit_indices
                        # print(cargs)
                        gate.qubit_indices=qargs
                        circuit.append(gate.instruction.operation, qargs, cargs)
                        # print(circuit.draw())
                    except:
                        print("Can't add all gates")
            else:
                #print(gate_qargs)
                circuit.swap(*gate_qargs)
        #print(circuit.draw())
        circuit = transpile(circuit, initial_layout=[bit_assignment_lp[l] for l in range(self._num_qubits)], optimization_level=0)
        # placeholder solution: remaining isolated single qubit gates are ignored
        return circuit


def build_problem_dict(gate_rep, topology, swap_time):
    problem_dict = {'problem':
                        {'gates': gate_rep.to_gates_json(),
                         'topology': topology,
                         'swap_time': swap_time}}
    return problem_dict
