import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models.centralities import getCentralityValuesDict
from models.cross_validation import rf_gridCV
from models.graph_generator import generateLargeGraphs, generateSmallGraphs
import json

from models.data_set import buildDataSet

# The sizes of graphs to be generated
from models.data_analysis import scatterPlotXDegreeSpread

# largeNs = [50]
#
# # The number of graphs of size N to be generated
# M = 50

# The features taken into account when training/testing
# features = ["closeness"]#["katz", "degree"]#, "eigenvector"]#, "degree", "pagerank"]
#
# # Spread parameters. Numbers for all IC params, None for WC.
# spread_params = [0.01]#, 0.01, None]

# Write to file?
wtf = False


def mainSmall(features, spread_prob, iterations, N):

    # Generate small graphs of size N from graph files
    graphs = generateSmallGraphs(N)

    # Machine learning
    result_dict = mainCompute(graphs, features, spread_prob, iterations)

    # Plot
    result_dict['small'] = True
    result_dict['N'] = N
    scatterPlotXDegreeSpread(result_dict)


def mainLarge(features, spread_prob, iterations, M, N):

    # Generate M graphs of size N
    graphs = generateLargeGraphs(M, N)

    # Machine learning
    result_dict = mainCompute(graphs, features, spread_prob, iterations)

    # Plot
    result_dict['small'] = False
    result_dict['N'] = N
    result_dict['M'] = M
    scatterPlotXDegreeSpread(result_dict)


def mainCompute(graphs, features, spread_prob, iterations):

    # Retrieve node centralities
    centralityDicts = getCentralityValuesDict(graphs, features)

    # Build data set
    X, y = buildDataSet(graphs, centralityDicts, spread_prob, iterations)

    # Train-test split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    #
    # # Scale
    # sc = StandardScaler()
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.transform(X_test)
    #
    # # print("Fitting RF using k=10 cross validation")
    # rf_gridCV.fit(X_train, y_train)
    #
    # # print("Scoring test set")
    score_test = score_training = 0# score_test = rf_gridCV.score(X_test, y_test)
    # score_training = rf_gridCV.score(X_train, y_train)
    #
    # print("Test score:", score_test)
    # print("Training score:", score_training)

    return {'X':X,
            'y':y,
            'spread_prob': spread_prob,
            'iterations': iterations,
            'features': features,
            'score_test': score_test,
            'score_training': score_training
            }


# def writeToFile(spread_param, N, M):
#     print("Writing to file")
#     with open('best_params_N' + str(N) + '_M' + str(M) + '_p' + str(spread_param) + '.txt', 'w') as outfile:
#         json.dump(rf_gridCV.best_params_, outfile)
#         outfile.write("\n\nBest estimator: " + str(rf_gridCV.best_estimator_))
#         outfile.write("\n\nBest score: " + str(rf_gridCV.best_score_))
