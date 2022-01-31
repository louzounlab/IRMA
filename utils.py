import networkx
import random
import matplotlib.pyplot as plt
import myQueue
# from shoval.our_embeddings_methods.static_embeddings import *
import math
import numpy as np


def graph_from_file(path):
    delimiter = " "
    graph = networkx.Graph()
    f = open(path, "r")
    line = f.readline()
    while line != '':
        edge = line.split("\n")[0].split(delimiter)
        graph.add_node(int(edge[0]))
        graph.add_node(int(edge[1]))
        graph.add_edge(int(edge[0]), int(edge[1]))
        line = f.readline()
    f.close()
    return graph


def generate_file_graphs(params):
    graph = graph_from_file(params["graphs_directory"] + params["file_graph_name"])
    s = params["graphs-overlap"]
    graph1, graph2 = remove_networkx_edges(graph, s)
    sources = choose_seeds_file_graph(graph, graph1, graph2, params["seed_size"])
    return print_graphs_info(graph, graph1, graph2, params, sources)


def create_proj(graph):
    A = StaticEmbeddings(name="moshe", G=graph, initial_size=20, initial_method="node2vec", method="OGRE", H=None,
                         dim=2)

    proj = A.list_dicts_embedding[0]
    new_proj = {}
    x = np.array([proj[node][0] for node in proj])
    y = np.array([proj[node][1] for node in proj])
    min0 = np.percentile(x, 5)
    max0 = np.percentile(x, 95)
    min1 = np.percentile(y, 5)
    max1 = np.percentile(y, 95)
    for node in proj:
        if proj[node][0] < min0 or proj[node][0] > max0 or proj[node][1] < min1 or proj[node][1] > max1:
            continue
        a = (proj[node][0] - min0) / (max0 - min0)
        b = (proj[node][1] - min1) / (max1 - min1)
        new_proj[node] = [a, b]
    return new_proj


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


def generate_graphs(params):
    # graph = networkx.powerlaw_cluster_graph(params["nodes"], m=int(params["avg_deg"] * 0.5), p=0.1)
    graph = networkx.erdos_renyi_graph(params["nodes"], params["avg_deg"] / params["nodes"])
    s = params["graphs-overlap"]
    graph1, graph2 = remove_networkx_edges(graph, s)
    sources = choose_seeds2(graph1, graph2, params["seed_size"])
    return print_graphs_info(graph, graph1, graph2, params, sources)


def remove_networkx_edges(graph, s):
    random.uniform(0, 1)
    two_graphs = [networkx.Graph(), networkx.Graph()]
    for edge in graph.edges:
        for new_graph in two_graphs:
            rand = random.uniform(0, 1)
            if rand < s:
                new_graph.add_node(int(edge[0]))
                new_graph.add_node(int(edge[1]))
                new_graph.add_edge(int(edge[0]), int(edge[1]))
    return two_graphs[0], two_graphs[1]


def example_candidates(graph1):
    candidates = {}
    for node in graph1:
        candidates[node] = [node + i for i in range(-10, 10)]
        return candidates


def choose_high_deg_sources(graph, graph1, graph2, percent):
    num = math.ceil(percent * len(graph.nodes)) if percent < 1 else percent
    queue = myQueue.MaxQueue()
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
    while len(group) < num:
        rand = int(random.uniform(0, len(graph.nodes)))
        source, i = 0, 0
        for source in graph.nodes:
            if i == rand:
                break
            i += 1
        if (source not in group) and (source in graph1.nodes) and (source in graph2.nodes):
            group.append(source)
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
