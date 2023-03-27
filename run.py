import os
import json
import pickle

from irma import Irma, utils

DIR = "./results_pkl"


def dump(obj, name):
    os.makedirs(DIR, exist_ok=True)
    with open(os.path.join(DIR, name), "wb") as f:
        pickle.dump(obj, f)


params = json.load(open('config.json'))

seed_sizes = [25, 50, 100, 200, 400]

graph_num = 1
f1_results = {seed: [] for seed in seed_sizes}
recall_results = {seed: [] for seed in seed_sizes}
precision_results = {seed: [] for seed in seed_sizes}
count_edges_results = {seed: [] for seed in seed_sizes}

for seed in seed_sizes:
    for _ in range(10):
        file_path = params["graphs_directory"] + params["file_graphs"][graph_num]
        overlap = 0.6
        graph1, graph2, sources, nodes_to_match = utils.generate_file_graphs(file_path, overlap, seed, nodes_overlap=0.7)

        candidates = {}

        myIRMA = Irma(graph1, graph2, sources, nodes_to_match, graph1s_candidates=candidates)
        myIRMA.run()

        f1_results[seed].append(myIRMA.f1_array)
        recall_results[seed].append(myIRMA.recall_array)
        precision_results[seed].append(myIRMA.precision_array)
        count_edges_results[seed].append(myIRMA.count_edges_array)


dump(f1_results, f"graph{graph_num}_f1.pkl")
dump(recall_results, f"graph{graph_num}_recall.pkl")
dump(count_edges_results, f"graph{graph_num}_count.pkl")
dump(precision_results, f"graph{graph_num}_precision.pkl")
