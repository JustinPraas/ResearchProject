import time
from collections import OrderedDict

from networkx import betweenness_centrality, closeness_centrality, degree_centrality, katz_centrality_numpy, \
    pagerank_numpy, eigenvector_centrality_numpy

from models.util import secondsToMinSec


def getCentralityValuesDict(graphs, centralities):
    startTotal = time.time()
    print("Building centrality dictionaries...", end=" ")
    result = OrderedDict()

    for c in centralities:
        result[c] = OrderedDict()

    if "betweenness" in centralities:
        start = time.time()
        for g in graphs:
            result["betweenness"][g] = betweenness_centrality(g, k=len(g.nodes)) # k <= N, higher k gives better approximation
        end = int(time.time())
        # print("Betweenness:", end - start, "seconds")

    if "closeness" in centralities:
        start = time.time()
        for g in graphs:
            result["closeness"][g] = closeness_centrality(g)
        end = int(time.time())
        # print("Closeness:", end - start, "seconds")

    if "katz" in centralities:
        start = time.time()
        for g in graphs:
            result["katz"][g] = katz_centrality_numpy(g, 0.1)
        end = int(time.time())
        # print("Katz:", end - start, "seconds")

    if "eigenvector" in centralities:
        start = time.time()
        for g in graphs:
            result["eigenvector"][g] = eigenvector_centrality_numpy(g, 0.1)
        end = int(time.time())
        # print("Eigenvector:", end - start, "seconds")

    if "pagerank" in centralities:
        start = time.time()
        for g in graphs:
            result["pagerank"][g] = pagerank_numpy(g, 0.85) # alpha damping factor is 0.85 by default
        end = int(time.time())
        # print("PageRank:", end - start, "seconds")

    if "degree" in centralities:
        start = time.time()
        for g in graphs:
            result["degree"][g] = degree_centrality(g) # is normalized
        end = int(time.time())
        # print("Degree:", end - start, "seconds")

    endTotal = int(time.time())
    print("Duration: %d minutes and %d seconds" % secondsToMinSec(endTotal - startTotal))

    return result
