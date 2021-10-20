import json
import pprint
import random
import utils
import myQueue
import time
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import threading
import functools


class Pair:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.recommends = 0
        self.log_degree_sum = 0
        self.degree_sum = 0

    def __key(self):
        return str(self.a) + ',' + str(self.b)

    def __eq__(self, other):
        return (self.a == other.a) and (self.b == other.b)

    def __hash__(self):
        return hash(self.__key())


class Plots:
    def __init__(self):
        self.success_rate = []
        self.correct_pairs_order = []
        self.wrong_pairs_order = []
        self.correct_map_degree = []
        self.wrong_map_degree = []
        self.correct_map_log_degree = []
        self.wrong_map_log_degree = []
        self.correct_map_recommends = []
        self.wrong_map_recommends = []
        self.correct_map_pos = []
        self.wrong_map_pos = []


def generic_plot(x, y, title, file_name):
    # plt.title(title, fontdict={'fontsize': 22, 'fontname': 'Times New Roman'})
    plt.xlabel(x, fontsize=18, fontname="Times New Roman")
    plt.ylabel(y, fontsize=18, fontname="Times New Roman")
    plt.legend(loc="best")
    # plt.show()
    plt.savefig(file_name, format='eps')
    plt.clf()


class IRMA:
    def __init__(self, graph1, graph2, sources, params, nodes_to_match):
        self.A, self.Z = set(), set()
        self.graph1 = graph1
        self.graph2 = graph2
        self.sources = sources
        self.params = params
        self.plots = Plots()
        self.queue = myQueue.MaxQueue()
        self.prev_queue = myQueue.MaxQueue()
        self.M, self.unM = {}, {}
        self.pairs_dic = {}
        self.nodes_to_match = nodes_to_match
        self.f1_array, self.recall_array, self.precision_array, self.count_edges_array = [], [], [], []

    def expand_when_stuck(self):
        if self.params["parallel"]:
            return self.expand_when_stuck_parallel()
        threshold = 1000
        count_loops = 0
        self.load_sources()
        while len(self.A) != 0:
            count_loops += 1
            print(f"len(A)= {len(self.A)}")
            self.spread_array_marks_parallel(list(self.A))
            while self.candidate_in_queues(threshold):
                pair = self.best_candidate()
                if pair.a not in self.M and pair.b not in self.unM:
                    self.add_pair_to_map(pair)
                    if pair not in self.Z:
                        self.spread_pair_marks(pair)
                        self.Z.add(pair)
            print(f"end of while {count_loops}. time: {time.time() - start}")
            self.update_A()
        return self.M

    def expand_when_stuck_parallel(self):
        last_maps = []
        threshold = 1000
        countLoops = 0
        self.load_sources()
        while len(self.A) != 0:
            countLoops += 1
            self.spread_array_marks_parallel(list(self.A))
            while self.candidate_in_queues(threshold) or len(last_maps) > 0:
                if not self.candidate_in_queues(threshold):
                    self.spread_array_marks_parallel(last_maps)
                    last_maps = []
                    continue
                pair = self.best_candidate()
                if pair.a not in self.M and pair.b not in self.unM:
                    self.add_pair_to_map(pair)
                    if pair not in self.Z:
                        last_maps.append(Pair(pair.a, pair.b))
                        self.Z.add(pair)
            print(f"end of while {countLoops}. time: {time.time() - start}")
            self.update_A()
        return self.M

    def repairing_iteration(self, noisy_loop):
        if self.params["parallel"]:
            return self.repairing_iteration_parallel(noisy_loop)
        time_start_loop = time.time()
        threshold = 0 if noisy_loop else 1000
        self.load_sources()
        self.spread_array_marks_parallel(list(self.A))
        while self.candidate_in_queues(threshold):
            pair = self.best_candidate()
            if pair.a not in self.M and pair.b not in self.unM:
                self.add_pair_to_map(pair)
                self.spread_pair_marks(pair)
        print(f"end of loop. time: {time.time() - start}. took {round(time.time() - time_start_loop, 1)} sec")
        return self.M

    def repairing_iteration_parallel(self, noisy_loop):
        time_start_loop = time.time()
        threshold = 0 if noisy_loop else 1000
        self.load_sources()
        while self.candidate_in_queues(threshold):
            pair = self.best_candidate()
            if pair.a not in self.M and pair.b not in self.unM:
                self.add_pair_to_map(pair)
        pairs_array = [Pair(a, self.M[a]) for a in self.M]
        self.pairs_dic = self.count_marks_parallel(pairs_array)
        print(f"end of loop. time: {time.time() - start}. took {round(time.time() - time_start_loop, 1)} sec")
        return self.M

    def spread_array_marks(self, pairs_array):
        for pair in pairs_array:
            self.spread_pair_marks(pair)

    def spread_array_marks_parallel(self, pairs_array):
        new_marks = self.count_marks_parallel(pairs_array)
        for node1 in new_marks:
            for node2 in new_marks[node1]:
                self.add_mark(Pair(node1, node2), new_marks[node1][node2].recommends)

    def count_marks_parallel(self, new_maps):
        num = self.params["threads"]
        results, threads = [0] * num, [0] * num
        maps = [[] for i in range(num)]
        # map_keys = [x for x in new_maps.keys()]
        for i in range(len(new_maps)):
            maps[i % num].append(new_maps[i])
        for i in range(num):
            threads[i] = threading.Thread(target=self.every_thread, args=(results, maps[i], i,))
            threads[i].start()
        for i in range(num):
            threads[i].join()
        allkey = functools.reduce(set.union, map(set, map(dict.keys, results)))

        new_pairs_dic = {}
        all_pairs_array = []
        for key in allkey:
            new_pairs_dic[key] = {}
            results_key = [dict_i[key] if key in dict_i else {} for dict_i in results]
            all_pairs_for_key = functools.reduce(set.union, map(set, map(dict.keys, results_key)))
            for pair in all_pairs_for_key:
                sum = 0
                for result in results:
                    if key in result:
                        if pair in result[key]:
                            sum += result[key][pair]
                my_pair = Pair(key, pair)
                my_pair.recommends = sum
                new_pairs_dic[key][pair] = my_pair
                delta_degree = abs(len(graph1[key]) - len(graph2[pair]))
                priority = 1000 * my_pair.recommends - delta_degree
                all_pairs_array.append((priority, my_pair))
        return new_pairs_dic

    def every_thread(self, results, chunk, thread_num):
        result = {}
        for pair in chunk:
            for graph1Neighbor in self.graph1[pair.a]:
                if graph1Neighbor not in result:
                    result[graph1Neighbor] = {}
                for graph2Neighbor in self.graph2[pair.b]:
                    if graph2Neighbor not in result[graph1Neighbor]:
                        result[graph1Neighbor][graph2Neighbor] = 0
                    result[graph1Neighbor][graph2Neighbor] += 1
        results[thread_num] = result

    def spread_pair_marks(self, recommends_pair):
        for graph1Neighbor in graph1[recommends_pair.a]:
            for graph2Neighbor in graph2[recommends_pair.b]:
                pair_to_update = Pair(graph1Neighbor, graph2Neighbor)
                self.add_mark(pair_to_update)

    def add_mark(self, pair, num=1):
        if pair.a not in self.pairs_dic:
            self.pairs_dic[pair.a] = {}
        if pair.b not in self.pairs_dic[pair.a]:
            self.pairs_dic[pair.a][pair.b] = pair
        else:
            pair = self.pairs_dic[pair.a][pair.b]
        pair.recommends += num

        if self.queue.contains(pair):
            self.queue.addToPriority(1000 * num, pair)
        else:
            simple_delta_deg = abs(len(self.graph1[pair.a]) - len(self.graph2[pair.b]))
            self.queue.push(1000 * num - simple_delta_deg, pair)

    def update_A(self):
        self.A = set()
        for a in self.M:
            b = self.M[a]
            for graph1Neighbor in self.graph1[a]:
                if graph1Neighbor not in self.M:
                    for graph2Neighbor in self.graph2[b]:
                        if graph2Neighbor not in self.unM:
                            if Pair(graph1Neighbor, graph2Neighbor) not in self.Z:
                                pair = Pair(graph1Neighbor, graph2Neighbor)
                                self.A.add(pair)
                                self.Z.add(pair)

    def load_sources(self):
        for source in self.sources:
            self.M[source] = source
            self.unM[source] = source
            pair = Pair(source, source)
            self.A.add(pair)
            self.Z.add(pair)

    def add_pair_to_map(self, pair):
        # self.update_plots(pair)
        self.M[pair.a] = pair.b
        self.unM[pair.b] = pair.a
        if len(self.M) % 2000 == 0:
            print("done: " + str(len(self.M)))

    def prepare_to_iteration(self):
        self.A, self.Z = set(), set()
        self.plots = Plots()
        self.queue = myQueue.MaxQueue()
        self.M, self.unM = {}, {}
        self.prev_queue = self.create_queue_by_pairs_dic(self.pairs_dic)
        self.pairs_dic = {}

    def create_queue_by_pairs_dic(self, pairs_dic):
        pairs_array = []
        for key1, dic in pairs_dic.items():
            for key2, pair in dic.items():
                delta_deg = abs(len(self.graph1[pair.a]) - len(self.graph2[pair.b]))
                priority = pair.recommends * 1000 - delta_deg
                pairs_array.append((priority, pair))
        return myQueue.MaxQueue(pairs_array)

    def candidate_in_queues(self, threshold):
        return ((not self.queue.isEmpty()) and self.queue.top()[0] > threshold) or (
                (not self.prev_queue.isEmpty()) and self.prev_queue.top()[0] > threshold)

    def best_candidate(self):
        if self.queue.isEmpty():
            pop = self.prev_queue.pop()
        elif self.prev_queue.isEmpty():
            pop = self.queue.pop()
        elif self.prev_queue.top()[0] > self.queue.top()[0]:
            pop = self.prev_queue.pop()
        else:
            pop = self.queue.pop()
        return pop[1]

    def update_plots(self, pair, pos=-1):
        val = 1 if pair.a == pair.b else 0
        if pos == -1:
            pos = len(self.M) - len(self.sources)
        self.plots.success_rate.append(val)
        self.initialize_pair_values(pair)

        if val == 1:
            self.plots.correct_pairs_order.append(pair)
            self.plots.correct_map_pos.append(pos)
            self.plots.correct_map_degree.append(pair.degree_sum / 2)
            self.plots.correct_map_log_degree.append(pair.log_degree_sum / 2)
            self.plots.correct_map_recommends.append(pair.recommends)
        else:
            self.plots.wrong_pairs_order.append(pair)
            self.plots.wrong_map_pos.append(pos)
            self.plots.wrong_map_degree.append(pair.degree_sum / 2)
            self.plots.wrong_map_log_degree.append(pair.log_degree_sum / 2)
            self.plots.wrong_map_recommends.append(pair.recommends)

    def initialize_pair_values(self, pair):
        pair.a_degree = len(self.graph1[pair.a])
        pair.b_degree = len(self.graph2[pair.b])
        log_degree_a = math.log(pair.a_degree + 1)
        log_degree_b = math.log(pair.b_degree + 1)
        pair.log_degree_sum = log_degree_a + log_degree_b
        pair.degree_sum = pair.a_degree + pair.b_degree

    def save_plots(self):
        directory = self.params["plot_dir"]
        smooth = self.params["smooth"]

        good_pos = self.plots.correct_map_pos[0:len(utils.smooth(self.plots.correct_map_pos, smooth))]
        smoothed_wrong_map_pos = utils.smooth(self.plots.wrong_map_pos, smooth, good_pos[-1])
        last_pos = len(smoothed_wrong_map_pos)
        bad_pos = self.plots.wrong_map_pos[0:last_pos]

        degs = [len(self.graph1[a]) for a in self.graph1]
        deg_counter = (np.array(degs).max() + 1) * [0]
        for deg in degs:
            deg_counter[deg] += 1
        for i in range(len(deg_counter)):
            deg_counter[i] /= len(self.graph1)

        plt.plot(deg_counter)
        x, y, title, file_name = "Degree", "Frequency ", "Degree distribution", directory + "Degree distribution"
        generic_plot(x, y, title, file_name)

        plt.plot(utils.smooth(self.plots.success_rate, 30))
        x, y, title, file_name = "Time", "Percents", "Precision (sliding window of 30)", directory + "success percent"
        generic_plot(x, y, title, file_name)

        plt.plot(good_pos, utils.smooth(self.plots.correct_map_log_degree, smooth), label='Correct pairs', alpha=0.7)
        plt.plot(bad_pos, utils.smooth(self.plots.wrong_map_log_degree, smooth), label='Wrong pairs', alpha=0.7)
        x, y, title, file_name = "Time", "Log degree", "Log degree of mapped pairs by order of insertion", directory + "Log degree"
        generic_plot(x, y, title, file_name)

        plt.plot(good_pos, utils.smooth(self.plots.correct_map_degree, smooth), label='Correct pairs', alpha=0.7)
        plt.plot(bad_pos, utils.smooth(self.plots.wrong_map_degree, smooth), label='Wrong pairs', alpha=0.7)
        x, y, title, file_name = "Time", "Degree", "Degree of mapped pairs by order of insertion", directory + "degree"
        generic_plot(x, y, title, file_name)

        log_correct_recommends = [math.log(a) for a in utils.smooth(self.plots.correct_map_recommends, smooth)]
        log_wrong_recommends = [math.log(a) for a in utils.smooth(self.plots.wrong_map_recommends, smooth)]
        plt.plot(good_pos, log_correct_recommends, label='correct pairs', alpha=0.7)
        plt.plot(bad_pos, log_wrong_recommends, label='wrong pairs', alpha=0.7)
        x, y, title, file_name = "Time", "Marks", "Marks of mapped pairs by order of insertion", directory + "Marks"
        generic_plot(x, y, title, file_name)

        correct_end_recommends = [pair.recommends for pair in self.plots.correct_pairs_order]
        wrong_end_recommends = [pair.recommends for pair in self.plots.wrong_pairs_order]
        log_correct = [math.log(a) for a in utils.smooth(correct_end_recommends, smooth)]
        log_wrong = [math.log(a) for a in utils.smooth(wrong_end_recommends, smooth)]
        plt.plot(good_pos, log_correct, label='correct pairs', alpha=0.7)
        plt.plot(bad_pos, log_wrong, label='wrong pairs', alpha=0.7)
        x, y, title, file_name = "Time", "Marks", "Marks of at the end by order of insertion", directory + "Marks_end"
        generic_plot(x, y, title, file_name)

    def evaluate(self):
        count_correct = 0
        for a in self.M:
            if a == self.M[a]:
                count_correct += 1
        precision = count_correct / len(self.M)
        recall = count_correct / self.nodes_to_match
        f1 = 2 * recall * precision / (recall + precision)

        count_edges = 0
        for v in self.graph1:
            if v not in self.M:
                continue
            for u in self.graph1[v]:
                if u in self.M and self.M[v] in self.graph2[self.M[u]]:
                    count_edges += 1

        self.f1_array.append(round(f1 * 100, 2))
        self.recall_array.append(round(recall * 100, 2))
        self.precision_array.append(round(precision * 100, 2))
        self.count_edges_array.append(count_edges)
        if params["evaluate_prints"]:
            print("**evaluate**")
            print(f"should match: {self.nodes_to_match}")
            print(f"tried: {len(self.M)}")
            print(f"correct: {count_correct}")
            print(f"precision: {round(100 * precision, 3)}")
            print(f"recall: {round(100 * recall, 3)}")
            print(f"f1: {round(100*f1, 3)}")
            print(f"num of correct edges: {count_edges}")
            if len(self.count_edges_array) > 1:
                print(f"ratio of edges: {round(self.count_edges_array[-1] / (1 + self.count_edges_array[-2]), 3)}")
            print("***************\n")
        return count_correct

    def draw_graph(self, loop, proj):
        for u in self.graph1:
            if u not in proj:
                continue
            if u not in self.M:
                color = 'gray'
            else:
                color = 'green' if self.M[u] == u else 'red'
            c = plt.Circle((proj[u][0], proj[u][1]), 0.015, color=color, alpha=0.5)
            plt.gca().add_patch(c)
            for v in self.graph1[u]:
                if v not in proj:
                    continue
                plt.plot([proj[u][0], proj[v][0]], [proj[u][1], proj[v][1]], color='gray', alpha=0.1)
        iteration = "ExpandWhenStuck" if loop == 0 else f"Iteration {loop}"
        plt.title(f"{iteration}, recall={self.recall_array[-1]}, precision={self.precision_array[-1]}")
        # plt.show()
        plt.savefig(f"visual_map{loop}")
        plt.clf()

    def run_IRMA(self):
        # proj = utils.create_proj(self.graph1)
        count_loops, count_reduce = 0, 1
        self.expand_when_stuck()
        self.evaluate()

        noisy_loop, reduce_stage, expand_stage = False, False, True
        while noisy_loop or expand_stage or reduce_stage:
            # self.draw_graph(count_loops, proj)
            if noisy_loop:
                print("noisy loop")
            print(f"loop = {count_loops}\n")
            self.prepare_to_iteration()
            self.repairing_iteration(noisy_loop)
            self.evaluate()
            count_loops += 1

            if expand_stage:
                if count_loops > 3 and self.count_edges_array[-1] < self.count_edges_array[-2] * 1.02:
                    expand_stage, noisy_loop = False, True
            elif noisy_loop:
                noisy_loop, reduce_stage, = False, True
            elif reduce_stage:
                if count_reduce > 3:
                    reduce_stage = False
                count_reduce += 1


params = json.load(open('config.json'))["gradualProb"]
# random.seed(1000)
# print(params["plotDir"])
# os.mkdir(params["plotDir"])
graph_num = params["correntGraph"]
# params["seed_size"] = params["relevantSeeds"][graph_num] #################
params["path"] = params["relevantGraphs"][graph_num]
original_seed = params["relevantSeeds"][params["correntGraph"]]  # params["seed_size"]
for seed in [1]:  # 1, 3, 5, 10,
    seed_f1, seed_recall, seed_precision, seed_count_edges, seed_diff_m = [], [], [], [], []
    params["seed_size"] = int(seed * original_seed)
    s_array = [0.6]
    for s in s_array:
        params["graphs-overlap"] = s
        graph1, graph2, sources, params, nodes_to_match = utils.generate_graphs(params)
        # graph1, graph2, sources, params, nodes_to_match = utils.generate_file_graphs(params)
        pprint.pprint(params)
        myIRMA = IRMA(graph1, graph2, sources, params, nodes_to_match)
        start = time.time()
        myIRMA.run_IRMA()

        print(f"S = {s}, seed={int(seed * original_seed)}:")
        print(myIRMA.f1_array)
        print(f"time: {str(time.time() - start)}\n\n")
        seed_count_edges.append(myIRMA.count_edges_array)
        seed_f1.append(myIRMA.f1_array)
        seed_recall.append(myIRMA.recall_array)
        seed_precision.append(myIRMA.precision_array)

    print(f"seed = {int(seed * original_seed)}:")
    for i in range(len(s_array)):
        print(f"\"{s_array[i]}\": [")
        print(str(seed_count_edges[i]) + ",")
        print(str(seed_precision[i]) + ",")
        print(str(seed_recall[i]) + ",")
        last_comma = "" if i == len(s_array) - 1 else ","
        print(str(seed_f1[i]) + "]" + last_comma)
        print("\n\n")
