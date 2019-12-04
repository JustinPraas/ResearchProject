from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models.centralities import getCentralityValuesDict
from models.cross_validation import rf_gridCV
from models.graph_generator import generateLargeGraphs, generateSmallGraphs
import json

import models.data_set

# The sizes of graphs to be generated
from models.data_analysis import scatterPlotXDegreeSpread, heatmap

# Write to file?
from models.learning_curve import do_plot_LC

wtf = False

doML = False

concurrent = False


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
    X, y = models.data_set.buildDataSet(graphs, centralityDicts, spread_prob, iterations)

    # Train-test split
    if doML:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        # Scale
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        # print("Fitting RF using k=10 cross validation")
        rf_gridCV.fit(X_train, y_train)

        # print("Scoring test set")
        score_test = rf_gridCV.score(X_test, y_test)
        score_training = rf_gridCV.score(X_train, y_train)

        print("Test score:", score_test)
        print("Training score:", score_training)

    # TODO: remove
    if not doML:
        score_test = 0
        score_training = 0

    return {'X':X,
            'y':y,
            'spread_prob': spread_prob,
            'iterations': iterations,
            'features': features,
            'score_test': score_test,
            'score_training': score_training
            }


def generateHeatmapForSmall(features, probs, Ns, iterations):
    result_data = []
    for n in Ns:
        # Generate small graphs of size N from graph files
        graphs = generateSmallGraphs(n)

        temp_data = []
        for p in probs:
            temp_data.append(mainCompute(graphs, features, p, iterations)['score_test'])

        result_data.append(temp_data)

    heatmap(result_data, "_".join(features), probs, Ns)
