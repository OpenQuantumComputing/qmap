import os

import networkx as nx
from qiskit_ibm_runtime import QiskitRuntimeService


def linear_coupling(n):
    hardware_graph = nx.Graph()
    for i in range(1, n):
        hardware_graph.add_edge("p" + str(i), "p" + str(i + 1))

    name = "linear_" + str(n)
    return hardware_graph, name


def ring_coupling(n):
    hardware_graph = nx.Graph()
    for i in range(1, n):
        hardware_graph.add_edge("p" + str(i), "p" + str(i + 1))
    hardware_graph.add_edge("p" + str(n), "p" + str(1))
    name = "ring_" + str(n)
    return hardware_graph, name


def grid_coupling(n, m):
    hardware_graph = nx.Graph()
    for i in range(1, n):
        for j in range(1, m):
            hardware_graph.add_edge(str(i) + "_" + str(j), str(i + 1) + "_" + str(j))
            hardware_graph.add_edge(str(i) + "_" + str(j), str(i) + "_" + str(j + 1))
    for i in range(1, n):
        hardware_graph.add_edge(str(i) + "_" + str(m), str(i + 1) + "_" + str(m))
    for j in range(1, m):
        hardware_graph.add_edge(str(n) + "_" + str(j), str(n) + "_" + str(j + 1))

    mapping = {
        old_label: "p" + str(new_label + 1)
        for new_label, old_label in enumerate(hardware_graph.nodes())
    }
    hardware_graph = nx.relabel_nodes(hardware_graph, mapping)
    name = "grid_" + str(n) + "_" + str(m)
    return hardware_graph, name


def Y_coupling(n):
    hardware_graph = nx.Graph()
    prev_node = ["p" + str(1), "p" + str(1), "p" + str(1)]
    for i in range(2, n + 1):
        branch = i % 3
        hardware_graph.add_edge(prev_node[branch], "p" + str(i))
        prev_node[branch] = "p" + str(i)
    name = "Y_" + str(n)
    return hardware_graph, name


def IBM_graph(name):
    token_IBM = os.environ.get("QISKIT_IBM_TOKEN")
    if not token_IBM:
        raise RuntimeError(
            "IBM_graph requires the QISKIT_IBM_TOKEN environment variable."
        )
    service = QiskitRuntimeService(channel="ibm_quantum", token=token_IBM)
    hardware_backend = service.backend(name)
    G = nx.Graph()
    G.add_edges_from(hardware_backend.coupling_map)

    mapping = {
        old_label: "p" + str(new_label + 1)
        for new_label, old_label in enumerate(G.nodes())
    }
    G = nx.relabel_nodes(G, mapping)
    return G, name
