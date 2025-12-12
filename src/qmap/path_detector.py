class MinPathDetector:
    """An algorithm that finds all minimal paths in an undirected graph of at least a minimal given length.
    A path is minimal if it does not visit a node more than once, and if the path can not be shortened be removing a set of the inne nodes.
    This means there can be no edge between to nodes in the path that are not neighbours in the path."""


    paths : list[list[list[list[int]]]]
    """The minimal paths to be returned from the current call to the algorithm.
    The level 0 (outermost) list groups the paths by their start nodes.
    The level 1 list groups the paths of the given start node by their end nodes.
    The level 2 list holds all the paths from the given start node to the given end node.
    The level 3 (innermost) list holds the nodes of the specific path in the order they are visited.
    So the first node in a path will always be the start node the path is grouped under,
    and the last node in a path will always be the end node the path is grouped under.
    """

    current_node_paths : list[list[list[int]]]
    """The paths found for the currently selected start nod in the algorithm."""

    current_path : list[int]
    """The current path being built. The algorithm will search for all possible completions of this path first."""

    min_path_length : int
    """The minimal length of the returned paths."""

    nmb_nodes: int
    """The number of nodes in the graph. It is assumed that the nodes in the graph are all integers from 0 to the maximal node number given in the input edges."""

    neighbours : list[list[int]]
    """The edge neighbours for each node."""

    def __init__(self) -> None:

        self.paths : list[list[list[list[int]]]] = None
        self.current_node_paths : list[list[list[int]]] = None
        self.current_path : list[int] = None
        self.min_path_length = 0
        self.nmb_nodes = 0
        self.neighbours : list[list[int]] = None

    def find_paths(self, edges : list[tuple[int, int]], min_path_length : int) -> list[list[list[list[int]]]]:
        """Finds all minimal paths in an undirected graph of at least a minimal given length.
        The returned list is first grouped by all start nodes, then by all end nodes, then all minimal paths are listed between the two nodes.
        Each path lists the visited nodes, including the start and end nodes it is grouped under.
        There is a list object for every possible selection of start and end nodes,
        but the list is empty if the nodes are the same or if there is no minimal path founds between the nodes.

        args:
            - edges 'list[tuple[int, int]]' The edges in the graph, each edge is given by the nodes it connects.
            - min_path_length 'int' The minimal length of the returned paths.

        returns:
            The minimal paths found.
        """

        # Build neighbour relations
        self.min_path_length = min_path_length
        self.nmb_nodes = 1 + max(max(p) for p in edges)
        self.neighbours = list([] for _ in range(self.nmb_nodes))

        for edge in edges:
            self.neighbours[edge[0]].append(edge[1])
            self.neighbours[edge[1]].append(edge[0])

        # Search for the paths
        self.paths = []
        self.current_path = []
        for n in range(self.nmb_nodes):

            # Search for the paths starting at node n.
            self.current_node_paths = list([] for _ in range(self.nmb_nodes))
            self._find_paths_from_node(n, -1, 0)
            self.paths.append(self.current_node_paths)

        return self.paths

    def _find_paths_from_node(self, node : int, parent : int, level : int) -> None:
        """Searches for all possible minimal paths starting with the path in current_path with the given node appended.
        Minimal paths of desired length are stored in current_node_paths.
        It is assumed that the node to be appended is not already in current_path.
        
        args:
            - node 'int' The next node to be appended to the graph.
            - parent 'int' Either the last node in current_path or -1 if current_path is empty.
            - level 'int' The length of the current path. This is also the position the input node will have in current_path after is has been appended.
        """

        # Append the node to the graph
        self.current_path.append(node)
        next_level = level + 1

        if all((n not in self.current_path) for n in self.neighbours[node] if n != parent):
            # The new node added does not have any neighbours in the current_path, exept parent (the previous node in the path). Therefore current_path is still minimal.

            # Store the path if it is long enough.
            if next_level >= self.min_path_length:
                self.current_node_paths[node].append(self.current_path.copy())

            # We can now continue the search reccursivel by adding any neighbour of the node that is not the parent,
            # since we know such neighbours do not exist in current_path.
            for n in self.neighbours[node]:
                if n != parent:
                    self._find_paths_from_node(n, node, next_level)

        # Remove the node from the graph
        self.current_path.pop(level)
