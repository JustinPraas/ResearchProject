from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from models.centralities import getCentralityValuesDict
from models.cross_validation import rf_gridCV, knn_gridCV
from models.graph_generator import generateLargeGraphs, generateSmallGraphs, generateEgoFB

import models.data_set

# The sizes of graphs to be generated
from models.data_analysis import scatterPlotXDegreeSpread, heatmap, heatmaps, plotLearningCurve, \
    mean_confidence_interval

wtf = False

doML = True

concurrent = True


def mainSmall(features, spread_prob, iterations, N, do_knn = False, k = 5):

    # Generate small graphs of size N from graph files
    graphs = generateSmallGraphs(N)

    # Machine learning
    result_dict = mainCompute(graphs, features, None, spread_prob, iterations, do_knn, k)

    # Plot
    result_dict['small'] = True
    result_dict['N'] = N
    result_dict['conf_interval'] = mean_confidence_interval(result_dict['y'])
    scatterPlotXDegreeSpread(result_dict)

    print(result_dict['conf_interval'])
    return result_dict


def mainLarge(features, spread_prob, iterations, M, N, do_knn = False, k = 5):

    # Generate M graphs of size N
    graphs = generateLargeGraphs(M, N)

    # Machine learning
    result_dict = mainCompute(graphs, features, None, spread_prob, iterations, do_knn, k)

    # Plot
    result_dict['small'] = False
    result_dict['N'] = N
    result_dict['M'] = M
    scatterPlotXDegreeSpread(result_dict)

    return result_dict


def mainCompute(graphs, features, centralityDicts, spread_prob, iterations, do_knn = False, k = 5):
    result_dict = {
            'spread_prob': spread_prob,
            'iterations': iterations,
            'features': features
    }

    if centralityDicts is None:
        centralityDicts = getCentralityValuesDict(graphs, features)

    # Build data set
    if concurrent:
        X, y = models.data_set.buildDataSetPar(graphs, centralityDicts, spread_prob, iterations)
    else:
        X, y = models.data_set.buildDataSet(graphs, centralityDicts, spread_prob, iterations)

    result_dict['X'] = X
    result_dict['y'] = y

    # Train-test split
    if doML:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        # Scale
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        if not do_knn:
            # print("Fitting RF using k=10 cross validation")
            rf_gridCV.fit(X_train, y_train)

            # print("Scoring test set")
            result_dict['RFR_R2_test'] = rf_gridCV.score(X_test, y_test)
            result_dict['RFR_R2_train'] = rf_gridCV.score(X_train, y_train)
        else:
            knn_gridCV.fit(X_train, y_train)

            y_pred = knn_gridCV.predict(X_test)
            result_dict['KNN_R2_test'] = r2_score(y_test, y_pred)

    return result_dict


def generateHeatmaps(features, feature_combs, probs, iterations, Ns, M = -1, do_knn = False, k = 5):
    result_data = {}
    meta_data = {
        'probs': probs,
        'Ns': Ns,
        'M': M,
        'do_knn': do_knn,
        'k': k,
    }

    for comb in feature_combs:
        result_data[comb] = []

    task_nr = 1
    no_tasks = len(Ns) * len(feature_combs) * len(probs)
    for n in Ns:
        if M == -1:
            # Generate small graphs of size N from graph files
            graphs = generateSmallGraphs(n)
        else:
            graphs = generateLargeGraphs(M, n)

        centralitiy_dicts = getCentralityValuesDict(graphs, features)

        for comb in feature_combs:
            temp_data = []
            for p in probs:
                print("Task %d/%d (N=%d, p=%s, comb=%s)" % (task_nr, no_tasks, n, str(p), str(comb)))
                if do_knn:
                    temp_data.append(mainCompute(graphs, features, centralitiy_dicts, p, iterations, do_knn, k)['KNN_R2_test'])
                else:
                    temp_data.append(mainCompute(graphs, features, centralitiy_dicts, p, iterations)['RFR_R2_test'])
                task_nr += 1

            result_data[comb].append(temp_data)

    heatmaps(result_data, meta_data)


def plotLC(features, M, N, spread_prob, iterations, sizes):
    data = mainLarge(features, spread_prob, iterations, M, N)
    plotLearningCurve(data['X'], data['y'], 10,  sizes)

#generateHeatmapsForSmall(["closeness", "pagerank", "degree"], [("closeness", "degree"), ("pagerank", "degree"), ("closeness"), ("pagerank")], [0.01, 0.02, 0.05], 1000, [6, 7, 8])
