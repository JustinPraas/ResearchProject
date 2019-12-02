import matplotlib.pyplot as plt

def printCentralityDicts(c_dicts):
    for c in c_dicts:
        for graph in c_dicts[c]:
            print("Graph", graph, "| nodes: ", end='')
            for n in graph.nodes:
                print("%d==%.2f, " % (n, c_dicts[c][graph][n]), end='')
            print("")
        print("")

def scatterPlot(X, y, gtitle="", xtitle="", ytitle=""):
    _, ydim = X.shape

    for i in range(0, ydim):
        Xs = [x[i] for x in X]
        plt.scatter(Xs, y, alpha=0.4, label=xtitle.split(", ")[i])
        i += 1

    plt.legend()
    plt.grid(True)
    plt.title(gtitle)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.show()