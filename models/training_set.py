import numpy as np

from models.information_diffusion import independentCascade


def buildTrainingSet(centrality_values, spreads):
    X = []
    y = []

    for centrality in centrality_values:
        temp_centrality = []

        for (g, node_centralities) in centrality_values[centrality]:

            for u in node_centralities:
                temp_centrality.append(node_centralities[u])
                y.append(spreads[(g, u)])

        X.append(temp_centrality)

    X = np.transpose(X)

    return X, y

def buildTrainingSet(graphs, centralities_dict):
    X, y = [], []

    for g in graphs:
        for u in g.nodes:
            temp_centralities = []
            spread = independentCascade(g, u, 0.1)

            for c in centralities_dict:
                temp_centralities.append(centralities_dict[c][g][u])

            X.append(temp_centralities)
            y.append(spread)

    return X, y
