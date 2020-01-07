from collections import OrderedDict

import networkit.centrality as nk

def getCentralityValuesDictPar(graph, centralities):
    print("Building centrality dictionaries... (parallel)", end=" ")
    result = OrderedDict()

    for c in centralities:
        result[c] = OrderedDict()

    result["betweenness"][graph] = nk.ApproxBetweenness(graph).run()

    result["closeness"][graph] = nk.ApproxCloseness(graph, len(graph.nodes())).run()

    result["katz"][graph] = nk.KatzCentrality(graph).run()

    result["eigenvector"][graph] = nk.EigenvectorCentrality(graph).run()

    result["pagerank"][graph] = nk.PageRank(graph, 0.85).run() # alpha damping factor is 0.85 by default

    result["degree"][graph] = nk.DegreeCentrality(graph, normalized=True).run() # is normalized

    return result