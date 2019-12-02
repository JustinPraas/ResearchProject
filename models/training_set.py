import time

import numpy as np

from models.information_diffusion import independentCascade, weightedCascade
from models.util import secondsToMinSec


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
                spread = independentCascade(graph, seed, spread_param, iterations)
            else:
                spread = weightedCascade(graph, seed, iterations)

            y.append(spread)

    Xn, yn = np.array(X), np.array(y)

    endTotal = int(time.time())
    print("Duration: %d minutes and %d seconds" % secondsToMinSec(endTotal - startTotal))

    return Xn, y
