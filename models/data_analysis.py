import matplotlib.pyplot as plt
import seaborn as sns
import pandas as ps
import numpy as np
import copy as cp

savePlot = True

sns.set()


def scatterPlotXDegreeSpread(result_dict):
    X, y = cp.copy(result_dict['X']), result_dict['y']
    N, features, spread_prob, iterations = \
        result_dict['N'], result_dict['features'], result_dict['spread_prob'], result_dict['iterations']

    f, ax = plt.subplots(figsize=(6.5, 6.5))
    sns.despine(f, left=True, bottom=True)

    palette = "RdYlBu"

    data = prepareData(X, features, y)

    cb = plt.cm.ScalarMappable(cmap=palette)
    sns.scatterplot(x=features[0], y="spread", hue="degree", data=data, ax=ax, marker="_", palette=palette)

    ax.get_legend().remove()
    ax.figure.colorbar(cb).set_label("Degree")

    if spread_prob:
        title = "IC, p = %.2f" % spread_prob
    else:
        title = "WC"
    title += ", spread iterations: %d" % iterations
    title = "N = %d, %s" % (N, title)

    plt.title(title)

    if savePlot:
        plt.savefig("./plots/scatter_N%d_p%s_it%d.png" %
                    (N, str(int(spread_prob*100)) if spread_prob else "WC", iterations),
                    bbox_inches='tight')

    plt.show()



def heatmap(data, title, xaxis_labels, yaxis_labels):
    df = ps.DataFrame(data)
    ax = sns.heatmap(df,
                     xticklabels=xaxis_labels,
                     yticklabels=yaxis_labels,
                     vmin=0, vmax=1,
                     cmap="YlGnBu",
                     square=True)

    ax.set(xlabel="p", ylabel="N")

    # Hack:
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)

    plt.title(title)

    if savePlot:
        plt.savefig("./plots/heatmap_%s_%s_%s.png" %
                    (title, "_".join(map(str, yaxis_labels)), "_".join(map(str, xaxis_labels))),
                    bbox_inches='tight')

    plt.show()


def prepareData(X, features, y):
    zipped = [np.append(a, b) for (a, b) in np.array(list(zip(X.tolist(), y)))]
    data = ps.DataFrame(zipped, columns=np.append(features, "spread"))
    return data



