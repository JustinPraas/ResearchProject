from collections import OrderedDict
from textwrap import wrap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from models.centralities import getCentralityValuesDict
from models.cross_validation import rf_gridCV, knn_gridCV
from models.graph_generator import generateLargeGraphs, generateSmallGraphs, generateEgoFB

import numpy as np
import pandas as ps

import models.data_set

# The sizes of graphs to be generated
from models.data_analysis import scatterPlotXDegreeSpread, heatmap, heatmaps, \
    mean_confidence_interval, makeScatterPlot, makeLearningCurve

wtf = False

doML = True

concurrent = True

centralities = ["degree", "betweenness", "closeness", "pagerank", "eigenvector", "katz"]

single_combs = [("degree"), ("betweenness"), ("closeness"), ("pagerank"), ("eigenvector"), ("katz")]


def mainSmall(features, spread_prob, iterations, N, do_knn=False, k=5):
    # Generate small graphs of size N from graph files
    graphs = generateSmallGraphs(N)

    # Machine learning
    result_dict = mainCompute(graphs, features, None, spread_prob, iterations, do_knn, k)

    # Plot
    result_dict['small'] = True
    result_dict['N'] = N
    result_dict['do_knn'] = do_knn
    result_dict['k'] = k
    result_dict['conf_interval'] = mean_confidence_interval(result_dict['y'])
    scatterPlotXDegreeSpread(result_dict)

    print(result_dict['conf_interval'])
    return result_dict


def mainLarge(features, spread_prob, iterations, M, N, do_knn=False, k=5):
    # Generate M graphs of size N
    graphs = generateLargeGraphs(M, N)

    # Machine learning
    result_dict = mainCompute(graphs, features, None, spread_prob, iterations, do_knn, k)

    # Plot
    result_dict['small'] = False
    result_dict['N'] = N
    result_dict['M'] = M
    result_dict['do_knn'] = do_knn
    result_dict['k'] = k
    scatterPlotXDegreeSpread(result_dict)

    return result_dict


def mainCompute(graphs, features, centralityDicts, spread_prob, iterations, do_knn=False, k=5):
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


def generateHeatmaps(features, feature_combs, probs, iterations, Ns, Ms=None, do_knn=False, k=5):
    result_data = {}

    for comb in feature_combs:
        result_data[comb] = []

    graphs = OrderedDict()
    centralities = OrderedDict()

    for n in Ns:
        if Ms is None:
            graphs[n] = generateSmallGraphs(n)
        else:
            n_index = Ns.index(n)
            graphs[n] = generateLargeGraphs(Ms[n_index], n)
        centralities[n] = getCentralityValuesDict(graphs[n], features)

    task_nr = 1
    no_tasks = len(Ns) * len(feature_combs) * len(probs)
    for comb in feature_combs:
        data = []
        for n in Ns:
            temp_data = []
            for p in probs:
                print("Task %d/%d (N=%d, p=%s, comb=%s)" % (task_nr, no_tasks, n, str(p), str(comb)))
                if do_knn:
                    temp_data.append(
                        mainCompute(graphs[n], features, centralities[n], p, iterations, do_knn, k)['KNN_R2_test'])
                else:
                    temp_data.append(mainCompute(graphs[n], features, centralities[n], p, iterations)['RFR_R2_test'])
                task_nr += 1
            data.append(temp_data)

        heatmap(data, probs, Ns, do_knn, comb, iterations)


def plotLC(features, M, N, spread_prob, iterations, steps):
    data = mainLarge(features, spread_prob, iterations, M, N)
    makeLearningCurve(data, features, 10, steps)


def saveDatasetWith(graphs, centrality_dicts, spread_prob, iterations, N, M=None, real=False):
    # Build data set
    X, y = models.data_set.buildDataSetPar(graphs, centrality_dicts, spread_prob, iterations)

    # Zip variables with label
    zipped = [np.append(a, b) for (a, b) in np.array(list(zip(X.tolist(), y)))]

    # Save the data to the file
    name = createFileName(M, N, iterations, real, spread_prob)
    np.savetxt("data/" + name + ".csv", zipped, delimiter=",")

    return zipped


def saveDataset(N, spread_prob, iterations, M=None, real=False):
    # Generate graphs and centrality dictionaries
    if M is not None:
        if N <= 10:
            raise Exception("Please enter a graph size > 10")
        graphs = generateLargeGraphs(M, N)
    elif real:
        graphs = [generateEgoFB()]
    else:
        graphs = generateSmallGraphs(N)

    centrality_dicts = getCentralityValuesDict(graphs, centralities)

    return saveDatasetWith(graphs, centrality_dicts, spread_prob, iterations, N, M, real)


def loadDataset(N, spread_prob, iterations, M=None, real=False):
    # Load the data to the file
    name = createFileName(M, N, iterations, real, spread_prob)

    try:
        # Try to load it from existing file
        zipped = np.loadtxt("data/" + name + ".csv", delimiter=',')
    except OSError:
        # Otherwise generate it and save it
        zipped = saveDataset(N, spread_prob, iterations, M, real)

    df = ps.DataFrame(zipped, columns=np.append(centralities, "spread"))

    # Add meta data
    df.prob = spread_prob
    df.N = N
    df.M = M
    df.real = real
    df.iterations = iterations

    return df


def scatterPlot(data_frame, features, save=False):
    # Generate plot
    plt = makeScatterPlot(data_frame, features)

    # Create title
    setPlotTitle(data_frame, plt)

    # Save if necessary
    if save:
        name = \
            createFileName(data_frame.M, data_frame.N, data_frame.iterations, data_frame.real, data_frame.spread_prob)
        plt.savefig("plots/scatter" + name + ".png", bbox_inches='tight')

    # Show plot
    plt.show()


def learningCurve(data_frame, features, steps, save=False):
    if data_frame['N'] < 10:
        raise Exception("It is disadviced to plot a learning curves for all non-isomorphic graphs")

    # Generate plot and corresponding data
    plt, data = makeLearningCurve(data_frame, features, 10, steps)

    # Create title
    setPlotTitle(data_frame, plt)

    # Save if necessary
    if save:
        name = \
            createFileName(data_frame.M, data_frame.N, data_frame.iterations, data_frame.real, data_frame.spread_prob)
        plt.savefig("plots/lc" + name + ".png", bbox_inches='tight')

    # Show plot
    plt.show()


def score(data_frame, features, knn=False):
    X = data_frame[features]
    y = data_frame["spread"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # Scale
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    if not knn:
        rf_gridCV.fit(X_train, y_train)
        return rf_gridCV.score(X_train, y_train), rf_gridCV.score(X_test, y_test)
    else:
        knn_gridCV.fit(X_train, y_train)
        y_pred = knn_gridCV.predict(X_test)
        return r2_score(y_test, y_pred)


'''
UTIL FUNCTIONS
'''


def setPlotTitle(data_frame, plt):
    prob_title_part = "WC" if data_frame.prob is None else "IC = %.3f" % data_frame.prob
    title = "Learning Curve: N = %d, %s, spread reps = %d" % (data_frame.N, prob_title_part, data_frame.iterations)
    plt.title("\n".join(wrap(title, 30)))


def createFileName(M, N, iterations, real, spread_prob):
    prob_string = "%.3f" % spread_prob if spread_prob is not None else "WC"
    M_string = "M%d_" % M if M is not None else ""
    real_string = "ego_FB" if real else ""
    N_string = real_string if real else N
    name = "N%d_%sP%s_IT%d.csv" % (N_string, M_string, prob_string, iterations)
    return name
