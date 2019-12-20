import math
from textwrap import wrap

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as ps
import numpy as np
import scipy.stats

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import learning_curve

from models.util import prepareData

savePlot = True

sns.set()


# IMPORTANT! INSPIRATION FROM: https://chrisalbon.com/machine_learning/model_evaluation/plot_the_learning_curve/
def makeLearningCurve(data_frame, features, cv, steps):
    X = data_frame[features]
    y = data_frame["spread"]

    train_sizes, train_scores, test_scores = learning_curve(RandomForestRegressor(n_estimators=100),
                                                            X,
                                                            y,
                                                            cv=cv,
                                                            scoring='r2',
                                                            n_jobs=-1,
                                                            train_sizes=np.linspace(0.01, 1.0, steps),
                                                            verbose=1)

    data = {'train_sizes': train_sizes,
            'train_scores': train_scores,
            'test_scores': test_scores}

    train_mean = data['train_mean'] = np.mean(train_scores, axis=1)
    train_std = data['train_std'] = np.std(train_scores, axis=1)

    test_mean = data['test_mean'] = np.mean(test_scores, axis=1)
    test_std = data['test_std'] = np.std(test_scores, axis=1)

    plt.figure(figsize=(4, 3))
    plt.ylim(0.5, 1)
    plt.plot(train_sizes, train_mean, '--', color="#f67f4b", label="Training score")
    plt.plot(train_sizes, test_mean, color="#36459c", label="Cross-validation score")

    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#feefa6")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#9dcee3")

    plt.xlabel("Training set size"), plt.ylabel("R²"), plt.legend(loc="best")

    return plt, data


def makeScatterPlot(data_frame, features):
    if len(features) > 2:
        raise Exception("Scatter can't have more than two features")

    f, ax = plt.subplots(figsize=(3, 2))
    sns.despine(f, left=True, bottom=True)
    palette = "RdYlBu"

    sns.scatterplot(x=features[0], hue=features[1], y="spread", data=data_frame, ax=ax, marker="_", palette=palette)

    ax.get_legend().remove()

    cb = plt.cm.ScalarMappable(cmap=palette)
    ax.figure.colorbar(cb).set_label(features[1])

    return plt


def makeScatterPlots(data_frame, features, hue):
    palette = "RdYlBu"
    fig, ax = plt.subplots(math.ceil(len(features) / 2), 2)

    for f in features:
        f_index = features.index(f)
        x_ax_index = f_index % 2
        y_ax_index = math.floor(f_index / 2)

        corr_ax = ax[y_ax_index][x_ax_index]
        sns.scatterplot(x=f, hue=hue, y="spread", data=data_frame, ax=corr_ax, marker="_", palette=palette)
        if x_ax_index == 1:
            corr_ax.set_ylabel('')
        corr_ax.get_legend().remove()

    plt.tight_layout()

    return plt


def createColorbarHor(hue="degree"):
    palette = "RdYlBu"
    fig, ax = plt.subplots()
    cb = plt.cm.ScalarMappable(cmap=palette)
    plt.gca().set_visible(False)
    fig.colorbar(cb, ax=ax, orientation="horizontal", fraction=.1).set_label(hue)

    plt.tight_layout()
    plt.savefig("plots/colorbar_" + hue + ".png")
    plt.show()


def makeHeatmaps(heatmap_data_sets):
    y_size = math.ceil(len(heatmap_data_sets) / 2)
    x_size = 2
    font_size = 10
    fontdict = {'fontsize': font_size}

    fig = plt.figure(figsize=(x_size * 2, y_size * 2))

    for i in range(0, len(heatmap_data_sets)):
        data_set = heatmap_data_sets[i]
        x_ax_index = i % 2
        y_ax_index = math.floor(i / 2)

        print(x_ax_index, y_ax_index)

        corr_ax = fig.add_subplot(y_size, x_size, i + 1)
        corr_ax.set_title("\n".join(wrap((", ").join(data_set.features), 15)), fontdict=fontdict)

        hm = sns.heatmap(data_set,
                         ax=corr_ax,
                         xticklabels=data_set.probs,
                         yticklabels=data_set.Ns,
                         vmin=0.75, vmax=1,
                         cmap="YlGnBu",
                         square=True,
                         cbar=False)

        hm.set_xticklabels(hm.get_xticklabels(), rotation=45)
        hm.set_yticklabels(hm.get_yticklabels(), rotation='horizontal')
        corr_ax.tick_params(axis='both', pad=-1, labelsize=font_size)

        show_xticklabels = x_ax_index == 0
        show_yticklables = y_ax_index == math.ceil(len(heatmap_data_sets) / 2) - 1

        if show_xticklabels:
            corr_ax.set_ylabel("N", rotation=0, labelpad=10, fontdict=fontdict, verticalalignment='center')

        if show_yticklables:
            corr_ax.set_xlabel("p", labelpad=6, fontdict=fontdict)

        # Hack:
        bottom, top = corr_ax.get_ylim()
        corr_ax.set_ylim(bottom + 0.5, top - 0.5)

    plt.subplots_adjust(wspace = -0.70)
    plt.tight_layout()

    return plt


def makeHeatmap(data, x_axis_labels, yaxis_labels):
    df = ps.DataFrame(data)

    x_axis_labels = [x if x is not None else "WC" for x in x_axis_labels]

    plt.figure(figsize=(4, 3))
    ax = sns.heatmap(df,
                     xticklabels=x_axis_labels,
                     yticklabels=yaxis_labels,
                     vmin=0.75, vmax=1,
                     cmap="YlGnBu",
                     square=True,
                     cbar_kws={'label': 'R²'})

    ax.set(xlabel="p", ylabel="N")

    # Hack:
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)

    plt.tight_layout()

    return plt, ax


# def makeHeatmaps(data, )


# TODO
# https://stackoverflow.com/a/15034143
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)

    # Calculate the mean and the standard error of the mean
    m, se = np.mean(a), scipy.stats.sem(a)

    #
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m - h, m, m + h
