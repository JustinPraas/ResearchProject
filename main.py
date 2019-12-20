from textwrap import wrap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from models.centralities import getCentralityValuesDict
from models.cross_validation import rf_gridCV, knn_gridCV
from models.data_set import buildDataSetPar, buildCentralityDataSet
from models.graph_generator import generateLargeGraphs, generateSmallGraphs, generateEgoFB

import numpy as np
import pandas as ps

from models.data_analysis import makeHeatmap, makeScatterPlot, makeLearningCurve, makeScatterPlots, createColorbarHor, \
    makeHeatmaps
from models.information_diffusion import independentCascadePar, weightedCascade, weightedCascadePar

wtf = False

doML = True

centralities = ["degree", "betweenness", "closeness", "pagerank", "eigenvector", "katz"]
single_combs = [("degree"), ("betweenness"), ("closeness"), ("pagerank"), ("eigenvector"), ("katz")]

datasets_path = "data/datasets/"
heatmap_data_path = "data/heatmap/"

def saveDatasetWith(graphs, centrality_dicts, spread_prob, iterations, N, M=None, real=False):
    # Save the data to the file
    name = createFileName(M, N, iterations, real, spread_prob)

    # Check if data set already exists
    try:
        # If so, return
        x = np.loadtxt(datasets_path + name + ".csv", delimiter=",")
        print("Dataset for %s already exists; Will not generate new one." % name)
        return x
    except OSError:
        # Build data set
        X, y = buildDataSetPar(graphs, centrality_dicts, spread_prob, iterations)

        # Zip variables with label
        zipped = [np.append(a, b) for (a, b) in np.array(list(zip(X.tolist(), y)))]

        np.savetxt(datasets_path + name + ".csv", zipped, delimiter=",")

        return zipped


def saveDataset(N, spread_prob, iterations, M=None, real=False):
    # Save the data to the file
    name = createFileName(M, N, iterations, real, spread_prob)

    # Check if data set already exists
    try:
        # If so, return
        x = np.loadtxt(datasets_path + name + ".csv", delimiter=",")
        print("Dataset for %s already exists; Will not generate new one." % name)
        return x
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
        zipped = np.loadtxt(datasets_path + name + ".csv", delimiter=',')
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


def scatterPlot(data_frame, features, hue=None, save=False):

    if hue is None:
        # Generate single plot with feature[1] as 'hue'
        plt = makeScatterPlot(data_frame, features)
    else:
        # Generate several plots in one with 'hue' as hue
        plt = makeScatterPlots(data_frame, features, hue)

    # Create title
    st = setPlotTitle(data_frame, plt)

    # Shift subplots down
    st.set_y(0.99)
    plt.gcf().subplots_adjust(top=0.93)

    # Save if necessary
    if save:
        name = \
            createFileName(data_frame.M, data_frame.N, data_frame.iterations, data_frame.real, data_frame.prob)
        name.replace("0.", "")
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


def saveHeatmapData(data_frame, features, Ns, probs, iterations, knn=False):
    name = createHeatmapFileName(features, Ns, probs, iterations, knn=knn)
    data_frame.to_csv("data/heatmap/" + name + ".csv", ",")

def loadHeatmapData(features, Ns, probs, iterations, knn=False):
    name = createHeatmapFileName(features, Ns, probs, iterations, knn=knn)
    df = ps.read_csv(heatmap_data_path + name + ".csv", ",", index_col=0)
    df.Ns = Ns
    df.probs = probs
    df.features = features
    df.iterations = iterations
    df.knn = knn

    return df


def heatmap(features, Ns, probs, iterations, Ms=None, knn=False, save=False):

    try:
        # Attempt to load an existing data file for these parameters
        df = loadHeatmapData(features, Ns, probs, iterations, knn=knn)
    except OSError:
        print("Creating new heatmap data for given parameters")
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

        df = ps.DataFrame(data, columns=probs, index=Ns)

    plt, _ = makeHeatmap(df, probs, Ns)

    # Create title
    setHeatmapTitle(features, iterations, knn, plt)

    # Save if necessary
    if save:
        name = createHeatmapFileName(features, Ns, probs, iterations, knn=knn)
        plt.savefig("plots/heatmap/" + name + ".png", bbox_inches='tight')
        saveHeatmapData(df, features, Ns, probs, iterations, knn=knn)

    # Show plot
    plt.show()

    return df


def heatmaps(feature_combs, Ns, probs, iterations, knn=False, save=True):
    data_sets = []
    for comb in feature_combs:
        data_sets.append(loadHeatmapData(comb, Ns, probs, iterations, knn=knn))

    plt = makeHeatmaps(data_sets)

    # Save if necessary
    if save:
        name = createHeatmapFileName([], Ns, probs, iterations, knn=knn)
        plt.savefig("plots/heatmap/" + name + "_combination.png", bbox_inches='tight')

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
FOR REAL, LARGE GRAPHS (N > 1000)
'''
def loadLargeDataset(hasSpread):
    try:
        X = np.loadtxt(datasets_path + "real_fb_ego.csv", delimiter=",")

        if hasSpread:
            df = ps.DataFrame(X, columns=np.append(centralities, "spread"))
        else:
            df = ps.DataFrame(X)

        # Add meta data
        df.hasSpread = hasSpread

        return df
    except OSError:
        print("Data set does not exist")


def saveLargeDatasetCentralities():
    graphs = [generateEgoFB()]
    X = buildCentralityDataSet(graphs, centralities)
    np.savetxt(datasets_path + "real_fb_ego.csv", X, delimiter=",")


def buildSpreadDatasetFromCentralityDataSet(graphs, prob, iterations):
    df = loadLargeDataset(False)

    y = []

    for g in graphs:
        for seed in g.nodes():
            if prob is not None:
                spread = independentCascadePar(g, seed, prob, iterations)
            else:
                spread = weightedCascadePar(g, seed, iterations)

            y.append(spread)

    df['spread'] = y

    prob_string = "WC" if prob is None else "%.3f" % prob
    df.to_csv(datasets_path + "real_fb_ego_P" + prob_string + ".csv")

    return df


'''
UTIL FUNCTIONS
'''
def setPlotTitle(data_frame, plt):
    prob_title_part = "WC" if data_frame.prob is None else "IC = %.3f" % data_frame.prob
    title = "N = %d, %s, spread reps = %d" % (data_frame.N, prob_title_part, data_frame.iterations)
    return plt.gcf().suptitle(title, fontsize=10)


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
