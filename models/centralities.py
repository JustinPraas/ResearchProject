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

        # print("Betweenness:", result["betweenness"])

    # if "closeness" in centralities:
    #     result["closeness"] = [(graphs.index(g), closeness_centrality(g)) for g in graphs]
    #     print("Closeness:", result["closeness"])
    #
    # if "katz" in centralities:
    #     result["katz"] = [(graphs.index(g), katz_centrality_numpy(g, 0.1)) for g in graphs]
    #     print("Katz:", result["katz"])
    #
    # if "eigenvector" in centralities:
    #     result["eigenvector"] = [(graphs.index(g), eigenvector_centrality_numpy(g, 0.1)) for g in graphs]
    #     print("EigenVector:", result["eigenvector"])
    #
    # if "pagerank" in centralities:
    #     result["pagerank"] = [(graphs.index(g), pagerank_numpy(g)) for g in graphs]
    #     print("PageRank:", result["pagerank"])
    #
    # if "degree" in centralities:
    #     result["degree"] = [(graphs.index(g), degree_centrality(g)) for g in graphs]
    #     print("Degree:", result["degree"])

    return result

# def getCentralityValues(centralities):
#
