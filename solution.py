from problem import Problem, ProblemGate


class SolutionJob:

    def __init__(self, start : int, duration : int, gate : ProblemGate, hardware_pair : tuple[str, str]) -> None:
        self.start = start
        self.duration = duration
        self.gate = gate
        self.hardware_pair = hardware_pair


    def __repr__(self):
        return str(self.to_json_dict())


    def to_json_dict(self) -> dict:
        return {"time": self.start, "gate": (None if self.gate == None else self.gate.id), "edge": [self.hardware_pair[0], self.hardware_pair[1]]}


def sol_job_compare(job1 : SolutionJob, job2 : SolutionJob) -> int:
    if job1.start != job2.start:
        return job1.start - job2.start
    elif job1.gate == None:
        if job2.gate == None:
            if job1.hardware_pair[0] < job2.hardware_pair[0]:
                return -1
            elif job2.hardware_pair[0] < job1.hardware_pair[0]:
                return 1
            elif job1.hardware_pair[1] < job2.hardware_pair[1]:
                return -1
            elif job2.hardware_pair[1] < job1.hardware_pair[1]:
                return 1
            else:
                return 0
        else:
            return 1
    else:
        if job2.gate == None:
            return -1
        else:
            return job1.gate.id - job2.gate.id


class Solution:

    def __init__(self, problem : Problem, optimality_ensured : bool, hardware_to_logical : dict[str, str], jobs : list[SolutionJob]) -> None:
        self.problem = problem
        self.optimality_ensured = optimality_ensured
        self.hardware_to_logical = hardware_to_logical
        self.jobs = jobs


    def to_json_dict(self) -> dict:
        operations = list(job.to_json_dict() for job in self.jobs)
        return {"bit_assignment": self.hardware_to_logical, "operations": operations}
    

    def debug_print_solution(self) -> None:
        print(f"Hardware to logical qubit: {self.hardware_to_logical}")
        print(f"Jobs:")
        for job in self.jobs:
            print(job)


    def depth(self) -> int:
        return max(job.start + job.duration for job in self.jobs)


    def swaps(self) -> int:
        return sum(job.gate == None for job in self.jobs)


    def validate(self, swap_duration : int) -> str:
        # Returns error message if validation failed, or None if validation did not fail

        result = self._validate_names()
        if result != None:
            return result
        result = self._validate_assignments()
        if result != None:
            return result
        result = self._validate_single_jobs(swap_duration)
        if result != None:
            return result
        result = self._validate_gates_covered()
        if result != None:
            return result
        result = self._validate_gates_order()
        if result != None:
            return result

        return None


    def _validate_names(self) -> str:
        # Returns error if worng names are used for logical qubit names or hardware qubit names
        hw_names = self.problem.hardware_qubit_names()
        lo_names = self.problem.logical_qubit_names()
        for hw_q, lo_q in self.hardware_to_logical.items():
            if not hw_q in hw_names:
                return f"Unknown hardware qubit name in assignments: {hw_q}"
            if not lo_q in lo_names:
                return f"Unknown logical qubit name in assignments: {lo_q}"
        return None


    def _validate_assignments(self) -> str:
        # Returns error if not every logical qubit is assigned to one and only one hardware qubit
        assigned_lo_names = list(self.hardware_to_logical.values())
        for lo_q in self.problem.logical_qubit_names():
            if not lo_q in assigned_lo_names:
                return f"Logical qubit {lo_q} is not assigned to any hardware qubit"
        if len(assigned_lo_names) != len(set(assigned_lo_names)):
            return f"Same logical qubit assigned to several hardware qubits"
        return None


    def _validate_single_jobs(self, swap_duration : int) -> str:
        # Returns error if there is any wrong setup in each single job: Start-time must be non-negative, duration must be non-negative and correct according to gate or swap duration, edge has expected format of existing hardware edge
        hw_edges = self.problem.hardware_edges + list((q2, q1) for q1, q2 in self.problem.hardware_edges)
        for job in self.jobs:
            if job.start < 0:
                return f"Job is scheduled at negative time: {job}"
            if job.duration <= 0:
                return f"Job has negative duration: {job}"
            if job.gate == None:
                if job.duration != swap_duration:
                    return f"Swap job duration is not same as common swap duration: {job}"
            else:
                if job.duration != job.gate.duration:
                    return f"Job and job gate have different duration: {job}"
            if not job.hardware_pair in hw_edges:
                return f"Job edge is not among hardware edges: {job}"
        return None


    def _validate_gates_covered(self) -> str:
        # Returns error if not every gate in problem is represented exactly once among the jobs
        nmb_gates = len(self.problem.gates)
        jobs_with_gates = list(job for job in self.jobs if job.gate != None)
        gates_covered = list(job.gate for job in jobs_with_gates)
        for job in jobs_with_gates:
            if not job.gate in self.problem.gates:
                return f"Job linked to unknown gate: {job}"
        for gate in self.problem.gates:
            if not gate in gates_covered:
                return f"Problem gate not covered: {gate}"
        if len(gates_covered) != len(set(gates_covered)):
            return f"Same gate covered by several jobs"
        
        return None


    def _validate_gates_order(self) -> str:
        # Returns error if the gates come in wrong order for each qubit, overlap in time, or if the logical qubits are not assigned to the hardware qubits when a gate is executed
        hw_to_logical = self.hardware_to_logical.copy()
        hw_time = {hw : 0 for hw in self.problem.hardware_qubit_names()}
        lo_gates = {lo : [] for lo in self.problem.logical_qubit_names()}
        lo_gate_pos = {lo : 0 for lo in self.problem.logical_qubit_names()}
        for gate in self.problem.gates:
            lo_gates[gate.pair[0]].append(gate)
            lo_gates[gate.pair[1]].append(gate)
        for job in self.jobs:
            hw0 = job.hardware_pair[0]
            hw1 = job.hardware_pair[1]
            lo0 = hw_to_logical[hw0] if hw0 in hw_to_logical else None
            lo1 = hw_to_logical[hw1] if hw1 in hw_to_logical else None
            gate = job.gate
            if gate == None:
                # Swap job
                if lo0 == None:
                    hw_to_logical.pop(hw1)
                else:
                    hw_to_logical[hw1] = lo0
                if lo1 == None:
                    hw_to_logical.pop(hw0)
                else:
                    hw_to_logical[hw0] = lo1
            else:
                # Gate job
                if (lo0 != gate.pair[0] or lo1 != gate.pair[1]) and (lo0 != gate.pair[1] or lo1 != gate.pair[0]):
                    return f"Current qubits at job edge are ({lo0}, {lo1}), does not fit with job gate: {job}"
                for lo in gate.pair:
                    if lo != None:
                        if lo_gates[lo][lo_gate_pos[lo]] != gate:
                            return f"Job gate is not next expected gate for {lo}: {job}"
                        lo_gate_pos[lo] += 1
            earliset_time = max(hw_time[hw0], hw_time[hw1])
            if job.start < earliset_time:
                return f"Job overlaps with previous job: {job}"
            hw_time[hw0] = earliset_time + job.duration
            hw_time[hw1] = earliset_time + job.duration

        return None
