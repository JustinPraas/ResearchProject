import time
from math import floor

import numpy as np
from joblib import Parallel, delayed
from multiprocessing import cpu_count

import main
from models.information_diffusion import independentCascade, weightedCascade, independentCascadePar
from models.util import secondsToMinSec, batches


def buildDataSet(graphs, centralities_dict, spread_param, iterations):
    X, y = [], []

    startTotal = time.time()
    print("Building data set...", end=" ")

    for graph in graphs:
        for seed in graph.nodes:
            temp_centralities = []

            for centr_key in centralities_dict:
                temp_centralities.append(centralities_dict[centr_key][graph][seed])

            X.append(temp_centralities)

            if spread_param is not None:
                if main.concurrent and iterations >= 1500 and len(graphs[0].nodes) > 50:
                    spread = independentCascadePar(graph, seed, spread_param, iterations)
                else:
                    spread = independentCascade(graph, seed, spread_param, iterations)
            else:
                spread = weightedCascade(graph, seed, iterations)

            y.append(spread)

    Xn, yn = np.array(X), np.array(y)

    endTotal = int(time.time())
    print("Duration: %d minutes and %d seconds" % secondsToMinSec(endTotal - startTotal))

    return Xn, y


def buildDataSetPar(graphs, centralities_dict, spread_param, iterations):
    X, y = [], []

    startTotal = time.time()
    print("Building data set...", end=" ")

    processors = cpu_count()
    bs = batches(graphs, floor(int(len(graphs)/processors)))
    print("Building dataset from", len(bs),
          "batches of", len(bs[0]), "graphs")
    results = Parallel(n_jobs=-1, verbose=1)(delayed(buildDataSetWorker)
                        (batch, centralities_dict, spread_param, iterations) for batch in bs)

    for _X, _y in results:
        for __X in _X:
            X.append(__X)

        y.extend(_y)

    Xn, yn = np.array(X), np.array(y)

    endTotal = int(time.time())
    print("Duration: %d minutes and %d seconds" % secondsToMinSec(endTotal - startTotal))

    return Xn, y

def buildDataSetWorker(graphs, centrality_dicts, spread_param, iterations):
    X, y = [], []

    for graph in graphs:
        for seed in graph.nodes:
            temp_centralities = []

            for centr_key in centrality_dicts:
                temp_centralities.append(centrality_dicts[centr_key][graph][seed])

            X.append(temp_centralities)

            if spread_param is not None:
                if main.concurrent and iterations >= 1500 and len(graphs[0].nodes) >= 200:
                    spread = independentCascadePar(graph, seed, spread_param, iterations)
                else:
                    spread = independentCascade(graph, seed, spread_param, iterations)
            else:
                spread = weightedCascade(graph, seed, iterations)

            y.append(spread)

    return (X, y)