import matplotlib.pyplot as plt


def scatterPlot(result_dict):
    X, y = result_dict['X'], result_dict['y']
    N, features, spread_prob = result_dict['N'], result_dict['features'], result_dict['spread_prob']

    for f in features:
        Xs = [x[features.index(f)] for x in X]
        plt.scatter(Xs, y, alpha=0.4, label=f)

    plt.legend()
    plt.grid(True)
    plt.title("N = %d, p = %.3f" % (N, spread_prob))
    plt.xlabel("Feature values")
    plt.ylabel("Estimated spread")
    plt.show()