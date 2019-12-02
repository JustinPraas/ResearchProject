import time

import numpy as np

from models.information_diffusion import independentCascade, weightedCascade


def buildDataSet(graphs, centralities_dict, spread_param, iterations):
    X, y = [], []

    start = time.time()
    for graph in graphs:
        for seed in graph.nodes:
            temp_centralities = []

            for centr_key in centralities_dict:
                temp_centralities.append(centralities_dict[centr_key][graph][seed])

            X.append(temp_centralities)

            if spread_param is not None:
                spread = independentCascade(graph, seed, spread_param, iterations)
            else:
                spread = weightedCascade(graph, seed, iterations)

            y.append(spread)
    end = time.time()

    Xn, yn = np.array(X), np.array(y)

    print("Duration", int(end - start), "seconds")
    return Xn, y
