class ProblemGate:

    def __init__(self, id : int, pair : tuple[str, str], duration : int) -> None:
        self.id = id
        self.pair = pair
        self.duration = duration

    def __repr__(self):
        return f"g{self.id}-(q{self.pair[0]},q{self.pair[1]})-{self.duration}"


class Problem:

    hardware_edges : list[tuple[str, str]]

    gates : list[ProblemGate]

    swap_time : int

    def __init__(self, problem_json) -> None:
        self.hardware_edges = [(edge[0], edge[1]) for edge in problem_json["topology"]]
        self.gates = []
        gate_id = 0
        for gate_json in problem_json["gates"]:
            q1 = gate_json["line1"]
            q2 = gate_json["line2"]
            duration = gate_json["duration"]
            self.gates.append(ProblemGate(gate_id, (q1, q2), duration))
            gate_id += 1
        self.swap_time = problem_json["swap_time"]


    def logical_qubit_names(self) -> list[str] :
        return list(sorted(set(q_name for gate in self.gates for q_name in gate.pair)))


    def hardware_qubit_names(self) -> list[str] :
        return list(sorted(set(q_name for pair in self.hardware_edges for q_name in pair)))


    def debug_print_problem(self) -> None:
        print("Hardware edges:")
        for hw_from, hw_to in self.hardware_edges:
            print(f"  {hw_from}-{hw_to}")
        print("Gates:")
        for gate in self.gates:
            print(gate)
