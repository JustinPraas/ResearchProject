from sklearn.model_selection import train_test_split

from models.centralities import getCentralityValuesDict
from models.cross_validation import rf_random
from models.graph_generator import generateLargeGraphs, generateGraphSeedPairs
from models.information_diffusion import *
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

from models.training_set import buildTrainingSet

largeNs = [100]
M = 50
features = ["betweenness", "closeness"]#, "katz", "eigenvector"]


def main(iterations):
    # Create data sets
    for n in largeNs:
        # Generate M graphs of size n

        print("Generating", n, "graphs")
        graphs = generateLargeGraphs(M, n)
        # print(M, "large graph" + ("s" if M > 1 else "") + " of size", n, "generated")

        # Generate graph-seed pairs for easier processing of other data
        graph_seed_pairs = generateGraphSeedPairs(graphs)

        # Apply spread models
        print("Applying spread models")
        ic_0_01_spreads = applySpread(graph_seed_pairs, independentCascade, 0.01)
        # ic_0_1_spreads = applySpread(graph_seed_pairs, independentCascade, 0.02)
        # wc_spreads = applySpread(graph_seed_pairs, weightedCascade)

        # Retrieve node centralities
        centralityDicts = getCentralityValuesDict(graphs, features)
        # print(centralityDicts)

        # Build training set
        print("Building training set")
        X, y = buildTrainingSet(graphs, centralityDicts)

        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

        print("Fitting RF using k=10 cross validation")
        rf_random.fit(X_train, y_train)

        print("Scoring test set")
        scores = rf_random.score(X_test, y_test)
        print(scores)

        print("Writing to file")
        with open('best_params.txt', 'w') as outfile:
            json.dump(rf_random.best_params_, outfile)
            outfile.write("\n\nBest estimator: " + str(rf_random.best_estimator_))
            outfile.write("\n\nBest score: " + str(rf_random.best_score_))
