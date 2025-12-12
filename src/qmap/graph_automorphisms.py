class GraphAutomorhisms:

    neighbours : list[list[bool]]

    edges : list[tuple[int, int]]

    automorphisms : list[list[int]]

    degrees : list[int]

    nmb_qubits : int

    non_isomorphic_edges : list[tuple[int, int]]

    def __init__(self, edges: list[tuple[int, int]]) -> None:

        self.nmb_qubits = 1 + max(max(e[0], e[1]) for e in edges)
        self.neighbours = [ [False] * self.nmb_qubits for _ in range(self.nmb_qubits) ]

        self.edges = []
        for edge in edges:
            if not self.neighbours[edge[0]][edge[1]]:
                self.neighbours[edge[0]][edge[1]] = True
                self.edges.append(edge)
            if not self.neighbours[edge[1]][edge[0]]:
                self.edges.append((edge[1], edge[0]))
                self.neighbours[edge[1]][edge[0]] = True

        self.degrees = [ sum(ngbh) for ngbh in self.neighbours ]
        self.automorphisms = None

    def get_edge_symmetries(self) -> None:

        self.automorphisms = []
        self._build_automorphisms({})

        self.non_isomorphic_edges = []
        nmb_edges = len(self.edges)
        edge_free = [True] * nmb_edges
        for idx in range(nmb_edges):
            if edge_free[idx]:
                edge = self.edges[idx]
                self.non_isomorphic_edges.append(edge)
                for aut in self.automorphisms:
                    perm_edge = (aut[edge[0]], aut[edge[1]])
                    idx2 = next(idx2 for idx2 in range(nmb_edges) if self.edges[idx2] == perm_edge)
                    edge_free[idx2] = False

    def _build_automorphisms(self, permutation : dict[int, int]) -> None:

        # print(f"Build automorphisms input permutation: {permutation}")
        unassigned_from = [q for q in range(self.nmb_qubits) if not q in permutation]
        if not unassigned_from:
            self.automorphisms.append([permutation[q] for q in range(self.nmb_qubits)])
            return

        assigned_from = sorted(permutation.keys())
        assigned_to = sorted(permutation.values())
        unassigned_to = [q for q in range(self.nmb_qubits) if not q in assigned_to]

        # For every qubit 'q' that is not a key in the permutation, the set of neighbours of 'q' that are keys in the permutation
        from_assigned_neighbours = { q: [ q_other for q_other in assigned_from if self.neighbours[q][q_other]] for q in unassigned_from }
        # For every qubit 'q' that is not a value in the permutation, the set of keys in the permutation that are mapped to neighbours of 'q'
        pulledback_to_assigned_neighbours = { q : [ q_other for q_other in assigned_from if self.neighbours[q][permutation[q_other]]] for q in unassigned_to }

        from_candidates = { q_to : [ q_from for q_from in unassigned_from if self.degrees[q_from] == self.degrees[q_to] and from_assigned_neighbours[q_from] == pulledback_to_assigned_neighbours[q_to] ] for q_to in unassigned_to }

        min_nmb_cand = min(len(cand) for cand in from_candidates.values())
        if min_nmb_cand == 1:
            singles = { q_to: cand[0] for q_to, cand in from_candidates.items() if len(cand) == 1 }
            if len(set(singles.values())) == len(singles):
                new_permutation = permutation.copy()
                for q_to, q_from in singles.items():
                    new_permutation[q_from] = q_to
                self._build_automorphisms(new_permutation)
        elif min_nmb_cand > 1:
            q_to, q_from_cand = next((q, cand) for q, cand in from_candidates.items() if len(cand) == min_nmb_cand)
            for q_from in q_from_cand:
                new_permutation = permutation.copy()
                new_permutation[q_from] = q_to
                self._build_automorphisms(new_permutation)

    def _build_automorphisms_X(self, unassigned_from : list[int], unassigned_to : list[int], assigned_to : list[int], inverse_assigned_permutation : dict[int, int]) -> None:

        from_neighbours = { q: [ q_other for q_other in assigned_to if self.neighbours[q][inverse_assigned_permutation[q_other]]] for q in unassigned_from }
        to_neighbours = { q: [ q_other for q_other in assigned_to if self.neighbours[q][q_other] ] for q in unassigned_to }

        candidates = { q: [ q_other for q_other in unassigned_to if self.degrees[q] == self.degrees[q_other] and from_neighbours[q] == to_neighbours[q_other] ] for q in unassigned_from }
        if not all(candidates.values()):
            return

        singles = { q: cand[0] for q, cand in candidates.items() if len(cand) == 1 }
        if singles:
            extra_assigned_to = list(singles.values())
            if len(extra_assigned_to) == len(set(extra_assigned_to)):
                new_inverse_assigned_perm = inverse_assigned_permutation.copy()
                new_unassigned_from = unassigned_from.copy()
                new_unassigned_to = unassigned_to.copy()
                new_assigned_to = unassigned_to.copy()
                for q_from, q_to in singles.items():
                    new_unassigned_from.remove(q_from)
                    new_unassigned_to.remove(q_to)
                    new_assigned_to.append(q_to)
                    new_inverse_assigned_perm[q_to] = q_from
                new_assigned_to.sort()
                self._build_symmetries(new_unassigned_from, new_unassigned_to, new_assigned_to, new_inverse_assigned_perm)
        else:
            best_nmb_cand = self.nmb_qubits + 1
            for q, cand in candidates.items():
                nmb_cand = len(cand)
                if nmb_cand < best_nmb_cand:
                    best_nmb_cand = nmb_cand
                    q_from = q
                    q_to_cand = cand
            for q_to in q_to_cand:
                pass

if __name__=="__main__":
    edges_sets = [
        [ (0, 1), (1, 2), (2, 3) ],   # Simple linear, odd length
        [ (0, 1), (1, 2), (2, 3), (3, 4) ],   # Simple linear, even length
        [ (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7) ],   # Longer linear
        [ (0, 1), (1, 2), (2, 3), (3, 4), (0, 4) ],   # Circular
        [ (0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5) ]   # Grid of 2x3 qubits with all shortest vertical+horizontal grid lines
    ]
    edges_sets_X = [
        [ (0, 1), (1, 2), (2, 3) ]
    ]
    print()
    for edges in edges_sets:
        gr_a = GraphAutomorhisms(edges)
        gr_a.get_edge_symmetries()
        print(f"Edges in = {edges}")
        print(f"Non-isomorphic candidates = {gr_a.non_isomorphic_edges}")
        print(f"Automorphisms = {gr_a.automorphisms}")
        print()
