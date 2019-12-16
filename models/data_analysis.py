from textwrap import wrap

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as ps
import numpy as np
import copy as cp
import scipy.stats

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import learning_curve

from models.util import prepareData

savePlot = True

sns.set()


# IMPORTANT! INSPIRATION FROM: https://chrisalbon.com/machine_learning/model_evaluation/plot_the_learning_curve/
def plotLearningCurve(result_dict, cv, steps):
    X, y = cp.copy(result_dict['X']), result_dict['y']
    N, features, spread_prob, iterations = \
        result_dict['N'], result_dict['features'], result_dict['spread_prob'], result_dict['iterations']

    train_sizes, train_scores, test_scores = learning_curve(RandomForestRegressor(n_estimators=100),
                                                            X,
                                                            y,
                                                            cv=cv,
                                                            scoring='r2',
                                                            n_jobs=-1,
                                                            train_sizes=np.linspace(0.01, 1.0, steps))

    data = {'train_sizes': train_sizes,
            'train_scores': train_scores,
            'test_scores': test_scores}

    train_mean = data['train_mean'] = np.mean(train_scores, axis=1)
    train_std = data['train_std'] = np.std(train_scores, axis=1)

    test_mean = data['test_mean'] = np.mean(test_scores, axis=1)
    test_std = data['test_std'] = np.std(test_scores, axis=1)

    plt.figure(figsize=(4, 3))
    plt.ylim(0.5, 1)
    plt.plot(train_sizes, train_mean, '--', color="#f67f4b",  label="Training score")
    plt.plot(train_sizes, test_mean, color="#36459c", label="Cross-validation score")

    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#feefa6")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#9dcee3")

    plt.title("Learning Curve")
    plt.xlabel("Training Set Size"), plt.ylabel("R²"), plt.legend(loc="best")

    plt.tight_layout()

    saveScatterOrLC(N, data, iterations, spread_prob, True)

    plt.show()


def scatterPlotXDegreeSpread(result_dict):
    X, y = cp.copy(result_dict['X']), result_dict['y']
    N, features, spread_prob, iterations = \
        result_dict['N'], result_dict['features'], result_dict['spread_prob'], result_dict['iterations']

    f, ax = plt.subplots(figsize=(4, 3))
    sns.despine(f, left=True, bottom=True)

    palette = "RdYlBu"

    data = prepareData(X, features, y)

    cb = plt.cm.ScalarMappable(cmap=palette)
    sns.scatterplot(x=features[0], y="spread", hue="degree", data=data, ax=ax, marker="_", palette=palette)

    ax.get_legend().remove()
    ax.figure.colorbar(cb).set_label("Degree")

    saveScatterOrLC(N, data, iterations, spread_prob, False)

    plt.show()


def saveScatterOrLC(N, data, iterations, spread_prob, LC):
    if spread_prob:
        title = "IC, p = %.2f" % spread_prob
    else:
        title = "WC"
    title += ", iter: %d" % iterations
    title = "N = %d, %s" % (N, title)
    plt.title(title)
    if savePlot:
        name = "./plots/scatter/%s" % ("scatter" if not LC else "LC")
        name += "_N%d_p%s_it%d" % (
        N, str(int(spread_prob * 100)) if spread_prob else "WC", iterations)
        plt.savefig(name + ".png", bbox_inches='tight')
        with open(name + '.txt', 'w') as outfile:
            outfile.write(title + ":\n")
            if not LC:
                outfile.write(data.to_string())
            else:
                outfile.write("Train sizes: " + str(data['train_sizes']))
                outfile.write("Train scores: " + str(data['train_scores']))
                outfile.write("Test scores: " + str(data['test_scores']))
                outfile.write("Train mean: " + str(data['train_mean']))
                outfile.write("Train std: " + str(data['train_std']))
                outfile.write("Test mean: " + str(data['test_mean']))
                outfile.write("Test std: " + str(data['test_std']))


def heatmaps(data, metadata):
    for comb in data.keys():
        title = "RFR: " if metadata['do_knn'] == False else "KNN: "
        title += "_".join([x for x in comb])

        heatmap(data[comb],
                title,
                metadata['probs'],
                metadata['Ns'])


def heatmap(data, xaxis_labels, yaxis_labels, do_knn, comb, iterations):
    df = ps.DataFrame(data)

    xaxis_labels = [x if x is not None else "WC" for x in xaxis_labels]

    plt.figure(figsize=(4, 3))
    ax = sns.heatmap(df,
                     xticklabels=xaxis_labels,
                     yticklabels=yaxis_labels,
                     vmin=0, vmax=1,
                     cmap="RdYlBu",
                     square=True,
                     cbar_kws={'label': 'R²'})

    ax.set(xlabel="p", ylabel="N")

    # Hack:
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)

    plt.tight_layout()

    title = "RFF: " if do_knn == False else "KNN: "
    combs = "_".join(comb)
    title += "%s, spread reps: %d" % (str(combs), iterations)
    plt.title("\n".join(wrap(title, 30)))

    if savePlot:
        name = "./plots/heatmap/%s" % "RFF" if not do_knn else "KNN"
        name += "_%s_%s_%s" % (combs, "_".join(map(str, yaxis_labels)), "_".join(map(str, xaxis_labels)))
        plt.savefig(name + ".png", bbox_inches='tight')
        with open(name + '.txt', 'w') as outfile:
            outfile.write(title + ":\n")
            outfile.write(df.to_string())

    plt.show()


# TODO
# https://stackoverflow.com/a/15034143
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)

    # Calculate the mean and the standard error of the mean
    m, se = np.mean(a), scipy.stats.sem(a)

    #
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m - h, m, m + h



