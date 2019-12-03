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


# def writeToFile(spread_param, N, M):
#     print("Writing to file")
#     with open('best_params_N' + str(N) + '_M' + str(M) + '_p' + str(spread_param) + '.txt', 'w') as outfile:
#         json.dump(rf_gridCV.best_params_, outfile)
#         outfile.write("\n\nBest estimator: " + str(rf_gridCV.best_estimator_))
#         outfile.write("\n\nBest score: " + str(rf_gridCV.best_score_))