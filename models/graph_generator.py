import time

from networkx import barabasi_albert_graph, from_graph6_bytes

import multiprocessing as mp


# def generateLargeGraphsPar(M, N):
#     output = mp.Queue()
#
#     results = []
#
#     processes = [mp.Process(target = generateLargeGraphs,
#                             args   = (N, output)) for x in range(M)]
#
#     for p in processes:
#         p.start()
#
#     while True:
#         try:
#             while not output.empty():
#                 result = output.get()
#                 results.append(result)
#         except output.Empty:
#             pass
#
#         allExited = True
#         for p in processes:
#             if p.exitcode is None:
#                 allExited = False
#                 break
#
#         if allExited & output.empty():
#             break
#         else:
#             time.sleep(0.3)
#
#     return results

# def generateLargeGraphs(N, output):
#     # Generate graphs using barabasi-albert
#     graphs = []
#     # numOfEdges = randrange(1, N)
#     num_of_edges = min(100, N//10)
#     # print("Edges: ", num_of_edges)
#     # for m in range(0, M):
#     output.put(barabasi_albert_graph(N, num_of_edges))
#         # m += 1
#
#     # print(M, "large graph" + ("s" if M > 1 else "") + " of size", n, "generated")
#
#     # output.put(graphs)

def generateLargeGraphs(M, N):
    start = int(time.time())

    # Generate graphs using barabasi-albert
    graphs = []
    num_of_edges = min(100, N // 10)
    for m in range(0, M):
        graphs.append(barabasi_albert_graph(N, num_of_edges))
        m += 1

    end = int(time.time())
    # print("Duration", end - start, "seconds")
    return graphs


def generateSmallGraphs(N):

    if not (6 < N < 10):
        raise Exception('The size for small graphs should be between 6 and 9')
        exit(0)

    start = int(time.time())

    result = []
    with open('./graphs/graph' + str(N) + 'c.g6') as graphs:
        graph_line = graphs.readline()
        while graph_line:
            graph_line = graph_line.rstrip().encode('ascii')
            graph = from_graph6_bytes(graph_line)
            result.append(graph)
            graph_line = graphs.readline()

    end = int(time.time())
    print("Duration", end - start, "seconds")

    return result
