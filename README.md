# Seeded Graph Matching


This repo contains the code for the paper IRMA - Iterative Repair For Graph Matching.  
The full paper can be found [here](https://arxiv.org/abs/2208.09164).

### How to run
* If you have two graphs and a seed, you may represent them as networkx graphs, and use Irma for matching:

```python
from irma.irma import IRMA

graph1 = "networkx graph"
graph2 = "networkx graph"
nodes_to_match = []  # Just for statistics
seed = []

myIRMA = IRMA(graph1, graph2, seed, nodes_to_match)
myIRMA.run_IRMA()

```

* If you have only one graph, you can simulate one more using the function generate_file_graphs in utils:

```python

from irma import utils

file_path = "./path_to_graph"
overlap = 0.6
seed_size = 100
delimiter = ","

graph1, graph2, seed, nodes_to_match = utils.generate_file_graphs(file_path, overlap, seed_size, delimiter=delimiter)
```




This project meant to enable restoring all experiments done in IRMA paper.

The code makes use of several packages as:
networkx, random, matplotlib, math, numpy, json, pprint, time, threading, functools.
All can be installed using pip.

1. myQueue.py is an implementation to priority-queue that used along the algorithm.

2. utils.py implements some function, mostly to initilize data sets for the algorithm based on config.json

3. data.7z need to be extracted such that 'data' directory is in the same directory as IRMA.py, and inside are 6 files.

4. config.json control several parameters for the algorithm:
   - nodes: determine the amount nodes in the source graph. used only when use_file_graph = False.
   - avg_deg: determine the amount edges in the source graph. used only when use_file_graph = False.
   - seed_size: determine the size of the seed to use.
   - graphs-overlap: determine the S used to sample graphs from the source graph (as explained in the paper)
   - smooth: used during research for pretty plots.
   - parallel: determine if use the parallel version.
   - threads: only relevant if parallel = True
   - evaluate_prints: control prints along IRMA's run.
   - use_file_graph: control if use one of the graphs in 'data' as a source or either use a fully simulated graphs.
   - graph_number: a value in range 0-5 to choose which graph to use among those in 'file_graphs' field.
   - file_graphs: DO NOT TOUCH. list of all graphs in 'data' director.
   - graphs_directory: DO NOT TOUCH. path to the file graphs.
   - file_graph_name: DO NOT TOUCH. used in the code to keep the file_graph's name.
   - plot_dir: used during research for printing plots.
   - draw_dir: the code enable to embed the graph and print it. currently the relevant code is in comment.

5. shoval.7z: If wants to use the ability of draw_dir , this file need to be extracted such that 'shoval' directory is in thr same directory as IRMA.py.
then remove the comment from the relevant import in utils.py and three lines in the 'run_IRMA' function.

6. IRMA.py is the main logic of our algorithm.
   It starts by generating the data graphs for the algorithm (lines 455-468)
   and then initilizes IRMA object and perform run_IRMA(). This function
   control the pipline of running first EWS and latter the repairing-iterations.
   notice that 'evaluate' function is called after each iteration (including EWS)
   to print the status of our current map. 
   The code prints by itself all measures that have been used in the paper. 
   the implementation of 'repairing iteration' is a bit complicated but there is no reason to fully understand it in order to run IRMA. 