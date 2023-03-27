import math
import random

import networkx

from irma.max_queue import MaxQueue


def graph_from_file(path, delimiter=" "):
    graph = networkx.Graph()
    with open(path, "r") as f:
        line = f.readline()
        while line:
            edge = line.strip("\n").split(delimiter)
            graph.add_edge(int(edge[0]), int(edge[1]))
            line = f.readline()
    return graph


def generate_file_graphs(graph_path: str, edges_overlap: float, seed_size: float, delimiter=" ", nodes_overlap=1):
    graph = graph_from_file(graph_path, delimiter=delimiter)
    graph1, graph2 = remove_networkx_edges(graph, edges_overlap, nodes_overlap)

    seed_nodes = choose_seeds_file_graph(graph, graph1, graph2, seed_size)

    nodes_to_match = 0
    degree_one_counter = 0
    for node in graph:
        if graph1.has_node(node) and graph2.has_node(node):
            nodes_to_match += 1
            if len(graph1[node]) == 1 or len(graph2[node]) == 1:
                degree_one_counter += 1

    return graph1, graph2, seed_nodes, nodes_to_match


# def create_proj(graph):
#     A = StaticEmbeddings(name="moshe", G=graph, initial_size=20, initial_method="node2vec", method="OGRE", H=None,
#                          dim=2)
#
#     proj = A.list_dicts_embedding[0]
#     new_proj = {}
#     x = np.array([proj[node][0] for node in proj])
#     y = np.array([proj[node][1] for node in proj])
#     min0 = np.percentile(x, 5)
#     max0 = np.percentile(x, 95)
#     min1 = np.percentile(y, 5)
#     max1 = np.percentile(y, 95)
#     for node in proj:
#         if proj[node][0] < min0 or proj[node][0] > max0 or proj[node][1] < min1 or proj[node][1] > max1:
#             continue
#         a = (proj[node][0] - min0) / (max0 - min0)
#         b = (proj[node][1] - min1) / (max1 - min1)
#         new_proj[node] = [a, b]
#     return new_proj


def print_graphs_info(graph, graph1, graph2, params, sources):
    nodes_to_match = 0
    degree_one_counter = 0
    for node in graph:
        if graph1.has_node(node) and graph2.has_node(node):
            nodes_to_match += 1
            if len(graph1[node]) == 1 or len(graph2[node]) == 1:
                degree_one_counter += 1
    print(f"all nodes: {len(graph.nodes)}")
    print(f"nodes to match {nodes_to_match}")
    print(f"nodes with degree 1 {degree_one_counter}")
    print("seed: " + str(len(sources)))
    params["nodes"] = len(graph1.nodes)
    params["avg_deg"] = 2 * len(graph1.edges()) / len(graph1.nodes)
    return graph1, graph2, sources, params, nodes_to_match


def generate_graphs(nodes, avg_deg, seed_size, edge_overlap, node_overlap):
    graph = networkx.erdos_renyi_graph(nodes, avg_deg / nodes)
    graph1, graph2 = remove_networkx_edges(graph, edge_overlap, node_overlap)
    seeds = choose_seeds2(graph1, graph2, seed_size)

    nodes_to_match = 0
    for node in graph:
        if graph1.has_node(node) and graph2.has_node(node):
            nodes_to_match += 1

    return graph1, graph2, seeds, nodes_to_match


def remove_networkx_edges(graph, s, t=1):
    """
    Takes a graph and probabilities and create two new samples graphs based on the original graph.
    Parameters
    ----------
    graph: networkx graph  to sample from.
    s: the probability of edge to be included in the sample graph (in case it's nodes in the graph)
    t: the probability of node to be included in the sample graph.

    Returns
    -------
    Two networkx graphs
    """
    graph1, graph2 = networkx.Graph(), networkx.Graph()

    if t == 1:
        for (u, v) in graph.edges:
            u, v = int(u), int(v)
            if random.uniform(0, 1) < s:
                graph1.add_edge(u, v)

            if random.uniform(0, 1) < s:
                graph2.add_edge(u, v)
    else:
        nodes1 = [int(node) for node in graph.nodes if random.uniform(0, 1) <= t]
        nodes2 = [int(node) for node in graph.nodes if random.uniform(0, 1) <= t]

        for (u, v) in graph.edges:
            u, v = int(u), int(v)
            if u in nodes1 and v in nodes1 and random.uniform(0, 1) < s:
                graph1.add_edge(u, v)

            if u in nodes2 and v in nodes2 and random.uniform(0, 1) < s:
                graph2.add_edge(u, v)

        del nodes1
        del nodes2

    return graph1, graph2

def example_candidates(graph1):
    candidates = {}
    for node in graph1:
        candidates[node] = [node + i for i in range(-10, 10)]
        return candidates


def choose_high_deg_sources(graph, graph1, graph2, percent):
    num = math.ceil(percent * len(graph.nodes)) if percent < 1 else percent
    queue = MaxQueue()
    for node in graph1:
        if node not in graph2:
            continue
        queue.push(len(graph1[node]) + len(graph2[node]), node)
    group = []
    while len(group) < num:
        group.append(queue.pop()[1])
    return group


def choose_seeds_file_graph(graph, graph1, graph2, percent):
    num = math.ceil(percent * len(graph.nodes)) if percent < 1 else percent
    group = []
    nodes = list(graph.nodes)
    while len(group) < num:
        rand = random.randint(0, len(nodes) - 1)
        node = nodes[rand]
        if (node not in group) and (node in graph1.nodes) and (node in graph2.nodes):
            group.append(node)

    return group


def choose_seeds1(graph, percent):
    num = math.ceil(percent * graph.GetNodes()) if percent < 1 else percent
    group = []
    while len(group) < num:
        r = graph.GetRndNId()
        if group.count(r) == 0:
            group.append(r)
    return group


def choose_seeds2(graph1, graph2, percent):
    num = math.ceil(percent * len(graph1.nodes)) if percent < 1 else percent
    percent = num / len(graph1.nodes)
    group = []
    for node in graph1.nodes:
        r = random.uniform(0, 1)
        if r < percent and node in graph2.nodes:
            group.append(node)
    return group


def smooth(array, window, compare=-1):
    if len(array) < window:
        return array
    window_sum = 0
    smooth_array = []
    for i in range(0, window):
        window_sum += array[i]
    for i in range(window, len(array)):
        smooth_array.append(window_sum / window)
        window_sum -= array[i - window]
        window_sum += array[i]
    if smooth_array[-1] < 0.9 * compare:
        return array
    return smooth_array
