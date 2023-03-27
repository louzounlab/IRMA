import logging
import sys
import time
from functools import partial
from multiprocessing import Pool

from irma.max_queue import MaxQueue

logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(message)s', level=logging.DEBUG)


def count_mark_process(lst, g1, g2):
    results_dict = {}
    for (a, b) in lst:
        for graph1Neighbor in g1[a]:
            for graph2Neighbor in g2[b]:
                p = (graph1Neighbor, graph2Neighbor)
                if p in results_dict:
                    results_dict[p] += 1
                else:
                    results_dict[p] = 1

    return results_dict


class Irma:
    def __init__(self, graph1, graph2, sources, nodes_to_match,
                 parallel=False,
                 max_queue=-1,
                 threads=5,
                 verbose=True):

        self.A, self.Z = set(), set()
        self.graph1 = graph1
        self.graph2 = graph2
        self.sources = sources

        self.parallel = parallel
        self.threads = threads

        self.start = time.time()
        self.verbose = verbose

        self.marks_dict = {}
        self.queue = MaxQueue(max_size=max_queue)
        self.prev_queue = None

        self.M, self.unM = {}, {}
        self.nodes_to_match = nodes_to_match
        self.f1_array, self.recall_array, self.precision_array, self.count_edges_array = [], [], [], []
        self.times, self.M_size = [], []


    def vprint(self, msg):
        if self.verbose:
            logging.info(msg)

    def add_mark(self, a, b, num=1):
        p = (a, b)
        if p in self.marks_dict:
            self.marks_dict[p] += num
        else:
            self.marks_dict[p] = num

        if self.queue.contains(p):
            self.queue.addToPriority(1000 * num, p)
        else:
            simple_delta_deg = abs(len(self.graph1[a]) - len(self.graph2[b]))
            self.queue.push(1000 * num - simple_delta_deg, p)

    def spread_marks(self, lst):
        for (a, b) in lst:
            self.Z.add((a, b))
            for graph1Neighbor in self.graph1[a]:
                for graph2Neighbor in self.graph2[b]:
                    self.add_mark(graph1Neighbor, graph2Neighbor)

    def expand_when_stuck(self):
        if self.parallel:
            return self.expand_when_stuck_parallel()
        threshold = 1000
        count_loops = 0
        self.load_sources()
        while self.A:
            count_loops += 1
            self.spread_marks(self.A)

            while not self.queue.isEmpty() and self.queue.top()[0] > threshold:
                p = (a, b) = self.queue.pop()[1]
                if a not in self.M and b not in self.unM:
                    self.add_pair_to_map(a, b)
                    if p not in self.Z:
                        self.spread_marks([p])

            self.update_A()
            self.vprint(f"end of while {count_loops}. time: {round(time.time() - self.start, 1)}")

        return self.M

    def spread_array_marks_parallel(self, pairs, insert_to_queue=True):
        # need to insert to Z, and to the Queue.
        num = self.threads
        # replace with split
        maps = [[] for _ in range(num)]
        for i in range(len(pairs)):
            maps[i % num].append(pairs[i])

        pool = Pool(processes=self.threads)
        results = pool.map(partial(count_mark_process, g1=self.graph1, g2=self.graph2), maps)

        self.Z = self.Z.union(pairs)

        if insert_to_queue:
            for dict_ in results:
                for pair, recommends in dict_.items():
                    self.add_mark(pair[0], pair[1], recommends)
        else:
            for dict_ in results:
                for pair, recommends in dict_.items():
                    if pair in self.marks_dict:
                        self.marks_dict[pair] += recommends
                    else:
                        self.marks_dict[pair] = recommends

    def expand_when_stuck_parallel(self):
        last_maps = []
        threshold = 1000
        count_loops = 0
        self.load_sources()
        while self.A:
            count_loops += 1

            self.spread_array_marks_parallel(list(self.A))

            while last_maps or (not self.queue.isEmpty() and self.queue.top()[0] > threshold):
                if self.queue.isEmpty() or self.queue.top()[0] <= threshold:
                    self.spread_array_marks_parallel(last_maps)
                    last_maps = []
                    continue

                p = (a, b) = self.queue.pop()[1]
                if a not in self.M and b not in self.unM:
                    self.add_pair_to_map(a, b)
                    if p not in self.Z:
                        last_maps.append(p)

            self.update_A()
            self.vprint(f"end of while {count_loops}. time: {round(time.time() - self.start, 1)}")

        return self.M

    def repairing_iteration(self, noisy_loop):
        if self.parallel:
            return self.repairing_iteration_parallel(noisy_loop)

        time_start_loop = time.time()
        threshold = 0 if noisy_loop else 1000
        self.load_sources()
        self.spread_marks(list(self.A))

        while self.candidate_in_queues(threshold):
            p = (a, b) = self.best_candidate()
            if a not in self.M and b not in self.unM:
                self.add_pair_to_map(a, b)
                self.spread_marks([p])

        self.vprint(f"end of loop. time: {round(time.time() - self.start, 1)}."
                    f" took {round(time.time() - time_start_loop, 1)} sec")
        return self.M

    def repairing_iteration_parallel(self, noisy_loop):
        time_start_loop = time.time()
        threshold = 0 if noisy_loop else 1000
        self.load_sources()
        while self.candidate_in_queues(threshold):
            a, b = self.best_candidate()
            if a not in self.M and b not in self.unM:
                self.add_pair_to_map(a, b)

        self.marks_dict.clear()
        self.spread_array_marks_parallel(list(self.M.items()), insert_to_queue=False)

        self.vprint(
            f"end of loop. time: {time.time() - self.start}. took {round(time.time() - time_start_loop, 1)} sec")
        return self.M

    def update_A(self):
        self.A.clear()
        # maybe this is all the pairs in the queue?
        for a, b in self.M.items():
            for graph1Neighbor in self.graph1[a]:
                if graph1Neighbor not in self.M:
                    for graph2Neighbor in self.graph2[b]:
                        if graph2Neighbor not in self.unM:
                            if (graph1Neighbor, graph2Neighbor) not in self.Z:
                                self.A.add((graph1Neighbor, graph2Neighbor))

    def load_sources(self):
        for source in self.sources:
            self.M[source] = source
            self.unM[source] = source
            self.A.add((source, source))

    def add_pair_to_map(self, a, b):
        self.M[a] = b
        self.unM[b] = a

    def prepare_to_iteration(self):
        self.A, self.Z = set(), set()
        del self.queue
        self.queue = MaxQueue()
        self.M, self.unM = {}, {}
        self.prev_queue = self.create_queue_by_marks_dict()
        self.marks_dict.clear()

    def create_queue_by_marks_dict(self):
        pairs_array = []
        for (a, b), marks in self.marks_dict.items():
            delta_deg = abs(len(self.graph1[a]) - len(self.graph2[b]))
            priority = marks * 1000 - delta_deg
            pairs_array.append((priority, (a, b)))
        return MaxQueue(pairs_array)

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

    def evaluate(self):
        count_correct = 0
        for a in self.M:
            if a == self.M[a]:
                count_correct += 1
        precision = count_correct / len(self.M)
        recall = count_correct / self.nodes_to_match
        f1 = 2 * recall * precision / (recall + precision)

        # weight
        count_edges = 0
        for v in self.graph1:
            if v not in self.M:
                continue
            # All (u,v) in E1  and v in M
            for u in self.graph1[v]:
                if u in self.M and self.M[v] in self.graph2[self.M[u]]:
                    count_edges += 1

        self.f1_array.append(round(f1 * 100, 2))
        self.recall_array.append(round(recall * 100, 2))
        self.precision_array.append(round(precision * 100, 2))
        self.count_edges_array.append(count_edges)
        self.M_size.append(len(self.M))

        self.vprint(
            "\n------------ evaluate ------------\n"
            f"Should match: {self.nodes_to_match}\n"
            f"Tried: {len(self.M)}\n"
            f"Correct: {count_correct}\n"
            f"Precision: {round(100 * precision, 3)}\n"
            f"Recall: {round(100 * recall, 3)}\n"
            f"F1: {round(100 * f1, 3)}\n"
            f"Num of correct edges: {count_edges}\n"
            f"----------------------------------"
        )

        return count_correct

    def run(self, allow_noisy_loops=True, noisy_delta=0.02, max_num_of_iterations=100,
            min_delta=-1):
        """

        Parameters
        ----------
        allow_noisy_loops - noisy loops is the exploring iteration of irma algorithms, iteration with threshold 1.
        noisy_delta - minimum increasing in the weight (percents) before the exploring iteration
        max_num_of_iterations - maximum number of iterations
        min_delta - minimum increasing in the weight (percents) before stopping the algorithm

        Returns None
        -------

        """
        # Runs ExpandWhenStuck algorithm
        start_time = time.time()
        self.expand_when_stuck()
        self.evaluate()
        self.times.append(time.time() - start_time)

        if max_num_of_iterations == 0:
            return

        count_loops, count_reduce = 0, 1
        noisy_loop, reduce_stage, expand_stage = False, False, True
        while noisy_loop or expand_stage or reduce_stage:
            self.vprint(f"Starts loop number {count_loops}." + " noisy loop." if noisy_loop else "")

            start_time = time.time()
            self.prepare_to_iteration()
            self.repairing_iteration(noisy_loop)
            self.evaluate()
            self.times.append(time.time() - start_time)
            count_loops += 1

            # Stopping conditions
            if count_loops >= max_num_of_iterations or \
                    self.count_edges_array[-1] < self.count_edges_array[-2] * (1 + min_delta):
                return

            if expand_stage:
                if count_loops > 3 and \
                        allow_noisy_loops and \
                        self.count_edges_array[-1] < self.count_edges_array[-2] * (1 + noisy_delta):
                    expand_stage, noisy_loop = False, True
            elif noisy_loop:
                noisy_loop, reduce_stage, = False, True
            elif reduce_stage:
                if count_reduce > 3:
                    reduce_stage = False
                count_reduce += 1
