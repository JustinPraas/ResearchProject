from textwrap import wrap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from models.centralities import getCentralityValuesDict
from models.cross_validation import rf_gridCV, knn_gridCV
from models.data_set import buildDataSetPar
from models.graph_generator import generateLargeGraphs, generateSmallGraphs, generateEgoFB

import numpy as np
import pandas as ps

from models.data_analysis import makeHeatmap, makeScatterPlot, makeLearningCurve

wtf = False

doML = True

centralities = ["degree", "betweenness", "closeness", "pagerank", "eigenvector", "katz"]
single_combs = [("degree"), ("betweenness"), ("closeness"), ("pagerank"), ("eigenvector"), ("katz")]


def saveDatasetWith(graphs, centrality_dicts, spread_prob, iterations, N, M=None, real=False):
    # Save the data to the file
    name = createFileName(M, N, iterations, real, spread_prob)

    # Check if data set already exists
    try:
        # If so, return
        print("Dataset for %s already exists; Will not generate new one." % name)
        np.loadtxt("data/" + name + ".csv", delimiter=",")
    except OSError:
        # Build data set
        X, y = buildDataSetPar(graphs, centrality_dicts, spread_prob, iterations)

        # Zip variables with label
        zipped = [np.append(a, b) for (a, b) in np.array(list(zip(X.tolist(), y)))]

        np.savetxt("data/" + name + ".csv", zipped, delimiter=",")

        return zipped


def saveDataset(N, spread_prob, iterations, M=None, real=False):
    # Save the data to the file
    name = createFileName(M, N, iterations, real, spread_prob)

    # Check if data set already exists
    try:
        # If so, return
        print("Dataset for %s already exists; Will not generate new one." % name)
        return np.loadtxt("data/" + name + ".csv", delimiter=",")
    except OSError:
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
            createFileName(data_frame.M, data_frame.N, data_frame.iterations, data_frame.real, data_frame.prob)
        plt.savefig("plots/scatter/" + name + ".png", bbox_inches='tight')

    # Show plot
    plt.show()


def learningCurve(data_frame, features, steps, save=False):
    if data_frame.N < 10:
        raise Exception("It is disadviced to plot learning curves for all non-isomorphic graphs")

    # Generate plot and corresponding data
    plt, data = makeLearningCurve(data_frame, features, 10, steps)

    # Create title
    setPlotTitle(data_frame, plt)

    # Save if necessary
    if save:
        name = \
            createFileName(data_frame.M, data_frame.N, data_frame.iterations, data_frame.real, data_frame.prob)
        plt.savefig("plots/lc/" + name + ".png", bbox_inches='tight')

    # Show plot
    plt.show()


def heatmap(features, Ns, probs, iterations, Ms=None, knn=False, save=False):
    data = []

    task_nr = 1
    no_tasks = len(Ns) * len(probs)
    for n in Ns:
        temp_data = []
        for p in probs:
            print("Task %d/%d (N=%d, p=%s, comb=%s)" % (task_nr, no_tasks, n, str(p), features))

            if Ms is not None:
                data_frame = loadDataset(n, p, iterations, Ms[Ns.index(n)])
            else:
                data_frame = loadDataset(n, p, iterations)

            temp_data.append(score(data_frame, features, knn))

            task_nr += 1
        data.append(temp_data)

    plt = makeHeatmap(data, probs, Ns, knn, features, iterations)

    # Create title
    setHeatmapTitle(features, iterations, knn, plt)

    # Save if necessary
    if save:
        name = createHeatmapFileName(features, Ns, probs, iterations, knn=knn)
        plt.savefig("plots/heatmap/" + name + ".png", bbox_inches='tight')

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
        return rf_gridCV.score(X_test, y_test)
    else:
        knn_gridCV.fit(X_train, y_train)
        y_pred = knn_gridCV.predict(X_test)
        return r2_score(y_test, y_pred)


def generateDatasets(Ns, probs, iterations, Ms=None, real=False):

    if real:
        graphs = [generateEgoFB()]
        centrality_dicts = getCentralityValuesDict(graphs, centralities)

        for i in iterations:
            for p in probs:
                saveDatasetWith(graphs, centrality_dicts, p, i, -1, real=True)
    else:
        for n in Ns:
            if Ms is not None:
                graphs = generateLargeGraphs(Ms[Ns.index(n)], n)
            else:
                graphs = generateSmallGraphs(n)

            centrality_dicts = getCentralityValuesDict(graphs, centralities)

            for i in iterations:
                for p in probs:
                    if Ms is not None:
                        saveDatasetWith(graphs, centrality_dicts, p, i, n, M=Ms[Ns.index(n)])
                    else:
                        saveDatasetWith(graphs, centrality_dicts, p, i, n)


'''
UTIL FUNCTIONS
'''
def setPlotTitle(data_frame, plt):
    prob_title_part = "WC" if data_frame.prob is None else "IC = %.3f" % data_frame.prob
    title = "N = %d, %s,\nspread reps = %d" % (data_frame.N, prob_title_part, data_frame.iterations)
    plt.title("\n".join(wrap(title, 30)))


def setHeatmapTitle(features, iterations, knn, plt):
    features_string = ", ".join(features)
    knn_string = "KNN" if knn else "RFF"

    title = "%s: %s, spread reps = %d" % (knn_string, features_string, iterations)
    plt.title("\n".join(wrap(title, 30)))


def createFileName(M, N, iterations, real, spread_prob):
    prob_string = "%.3f" % spread_prob if spread_prob is not None else "WC"
    M_string = "M%s_" % str(M) if M is not None else ""
    real_string = "ego_FB" if real else ""
    N_string = real_string if real else N
    name = "N%d_%sP%s_IT%d" % (N_string, M_string, prob_string, iterations)
    return name


def createHeatmapFileName(features, Ns, probs, iterations, knn=False):
    Ns_string = "Ns" + "_".join([str(n) for n in Ns])

    probs_formatted = ["%.3f" % prob if prob is not None else "WC" for prob in probs]
    probs_string = "Ps" + "_".join(probs_formatted)

    features_formatted = ["%3s" % f for f in features]
    features_string = "_".join(features_formatted)

    knn_string = "KNN" if knn else "RFF"

    return "%s_%s_%s_IT%d_%s" % (knn_string, Ns_string, probs_string, iterations, features_string)
