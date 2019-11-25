from collections import OrderedDict

from networkx import betweenness_centrality, closeness_centrality, degree_centrality, katz_centrality_numpy, \
    pagerank_numpy, eigenvector_centrality_numpy


def getCentralityValuesDict(graphs, centralities):
    result = OrderedDict()

    for c in centralities:
        result[c] = OrderedDict()

    if "betweenness" in centralities:
        for g in graphs:
            result["betweenness"][g] = betweenness_centrality(g, normalized=True, k=len(g.nodes))

    if "closeness" in centralities:
        for g in graphs:
            result["closeness"][g] = closeness_centrality(g)

    if "katz" in centralities:
        for g in graphs:
            result["katz"][g] = katz_centrality_numpy(g, 0.1)

    if "eigenvector" in centralities:
        for g in graphs:
            result["eigenvector"][g] = eigenvector_centrality_numpy(g, 0.1)

    if "pagerank" in centralities:
        for g in graphs:
            result["pagerank"][g] = pagerank_numpy(g)

    if "degree" in centralities:
        for g in graphs:
            result["degree"][g] = degree_centrality(g)

    return result
