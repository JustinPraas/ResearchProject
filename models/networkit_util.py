from collections import OrderedDict

import networkit.centrality as nk

def getCentralityValuesDictPar(graphs, centralities):
    print("Building centrality dictionaries... (parallel)", end=" ")
    result = OrderedDict()

    for c in centralities:
        result[c] = OrderedDict()

    for g in graphs:
        result["betweenness"][g] = nk.ApproxBetweenness(g)

    for g in graphs:
        result["closeness"][g] = nk.ApproxCloseness(g, len(g))

    if "katz" in centralities:
        result["katz"][g] = nk.KatzCentrality(g)

    for g in graphs:
        result["eigenvector"][g] = nk.EigenvectorCentrality(g)

    for g in graphs:
        result["pagerank"][g] = nk.PageRank(g, 0.85) # alpha damping factor is 0.85 by default

    for g in graphs:
        result["degree"][g] = nk.DegreeCentrality(g) # is normalized

    return result