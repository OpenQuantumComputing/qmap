from problem_encoding import ProblemEncoding
from path_detector import MinPathDetector

class HardwareDistance:
    """Measures on distances in a hardware graph."""

    encoding : ProblemEncoding
    """The encoding of the input problem."""

    swap_distance : list[list[int]]
    """one less than the distance between two hardware qubits in the hardware graph. The first index is the index of the first qubit, the second index is the index of the second qubit.
    If the two qubits are the same, -1 is returned. If the qubits are neighbours on the hardware graph, 0 is returned. If the qubits are not neighbours but have a common neighbour, 1 is returned, etc."""

    hardware_paths : list[list[list[list[int]]]]
    """The minimal paths of length at least 3 in the hardware graph between two qubits
    The level 0 (outermost) list groups the paths by their start qubits.
    The level 1 list groups the paths of the given start qubit by their end qubits.
    The level 2 list holds all the paths from the given start qubit to the given end qubit.
    The level 3 (innermost) list holds the qubits of the specific path in the order they are visited.
    The first qubit in a path will always be the start qubit the path is grouped under,
    and the last qubit in a path will always be the end qubit the path is grouped under.
    """

    def __init__(self, encoding : ProblemEncoding, build_hardware_paths : bool) -> None:

        self.encoding = encoding
        self.swap_distance = self._build_swap_distances()
        self.hardware_paths = self._build_hardware_paths() if build_hardware_paths else None


    def _build_swap_distances(self) -> list[list[int]]:

        big = self.encoding.nmb_hardware_qubits * 2
        dist = [[-1 if i == j else big for j in range(self.encoding.nmb_hardware_qubits)] for i in range(self.encoding.nmb_hardware_qubits)]
        neigbh = [[] for _ in range(self.encoding.nmb_hardware_qubits)]
        for q1, q2 in self.encoding.hardware_edges:
            neigbh[q1].append(q2)
            neigbh[q2].append(q1)
        new_dist = [(i, i) for i in range(self.encoding.nmb_hardware_qubits)]
        while new_dist:
            q1, q2 = new_dist.pop(0)
            d = dist[q1][q2] + 1
            for q3 in neigbh[q2]:
                if dist[q1][q3] == big:
                    dist[q1][q3] = d
                    dist[q3][q1] = d
                    new_dist.append((min(q1, q3), max(q1, q3)))
            for q3 in neigbh[q1]:
                if dist[q2][q3] == big:
                    dist[q2][q3] = d
                    dist[q3][q2] = d
                    new_dist.append((min(q2, q3), max(q2, q3)))
        return dist


    def _build_hardware_paths(self) -> list[list[list[list[int]]]]:

        detector = MinPathDetector()
        return detector.find_paths(self.encoding.hardware_edges, 3)
