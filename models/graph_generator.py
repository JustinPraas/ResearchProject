from networkx import barabasi_albert_graph
from random import randrange


def generateLargeGraphs(M, N):
    # Generate graphs using barabasi-albert
    graphs = []
    # numOfEdges = randrange(1, N)
    num_of_edges = min(100, N//10)
    # print("Edges: ", num_of_edges)
    for m in range(0, M):
        graphs.append(barabasi_albert_graph(N, num_of_edges))
        m += 1

    return graphs


def generateGraphSeedPairs(graphs):
    result = []
    for g in graphs:
        for n in g.nodes:
            result.append((g, n))
    return result
