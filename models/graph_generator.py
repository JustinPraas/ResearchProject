import time

from networkx import barabasi_albert_graph, from_graph6_bytes, read_edgelist
from models.util import secondsToMinSec


def generateLargeGraphs(M, N):
    startTotal = time.time()
    print("Generating %d graphs of size %d..." % (M, N), end=" ")

    # Generate graphs using barabasi-albert
    graphs = []
    num_of_edges = min(100, N // 10)
    for m in range(0, M):
        graphs.append(barabasi_albert_graph(N, num_of_edges))
        m += 1

    endTotal = int(time.time())
    print("Duration: %d minutes and %d seconds" % secondsToMinSec(endTotal - startTotal))
    return graphs


def generateSmallGraphs(N):
    startTotal = time.time()
    print("Importing graphs of size %d..." % N, end=" ")

    if not (5 < N < 11):
        raise Exception('The size for small graphs should be between 6 and 10')
        exit(0)

    graphs = []
    with open('./graphs/non-isomorphs/graph' + str(N) + 'c.g6') as graphs_file:
        graph_line = graphs_file.readline()
        while graph_line:
            graph_line = graph_line.rstrip().encode('ascii')
            graph = from_graph6_bytes(graph_line)
            graphs.append(graph)
            graph_line = graphs_file.readline()

    endTotal = int(time.time())
    print("Duration: %d minutes and %d seconds" % secondsToMinSec(endTotal - startTotal))

    return graphs

def generateEgoFB():
    return read_edgelist('./graphs/facebook_egos')