def printCentralityDicts(c_dicts):
    for c in c_dicts:
        for graph in c_dicts[c]:
            print("Graph", graph, "| nodes: ", end='')
            for n in graph.nodes:
                print("%d==%.2f, " % (n, c_dicts[c][graph][n]), end='')
            print("")
        print("")


def secondsToMinSec(seconds):
    sec = int(seconds)
    mins = sec // 60
    remSec = sec - (60 * mins)
    return (mins, remSec)