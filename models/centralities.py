import time
from collections import OrderedDict

from networkx import betweenness_centrality, closeness_centrality, degree_centrality, katz_centrality_numpy, \
    pagerank_numpy, eigenvector_centrality_numpy

from models.util import secondsToMinSec, batches


def getCentralityValuesDict(graphs, centralities):
    startTotal = time.time()
    print("Building centrality dictionaries...", end=" ")
    result = OrderedDict()

    for c in centralities:
        result[c] = OrderedDict()

    if "betweenness" in centralities:
        for g in graphs:
            result["betweenness"][g] = betweenness_centrality(g, k=len(g.nodes))

    if "closeness" in centralities:
        for g in graphs:
            result["closeness"][g] = closeness_centrality(g)

    if "katz" in centralities:
        for g in graphs:
            result["katz"][g] = katz_centrality_numpy(g)

    if "eigenvector" in centralities:
        for g in graphs:
            result["eigenvector"][g] = eigenvector_centrality_numpy(g)

    if "pagerank" in centralities:
        for g in graphs:
            result["pagerank"][g] = pagerank_numpy(g)

    if "degree" in centralities:
        for g in graphs:
            result["degree"][g] = degree_centrality(g) # is normalized

    endTotal = int(time.time())
    print("Duration: %d minutes and %d seconds" % secondsToMinSec(endTotal - startTotal))

    return result