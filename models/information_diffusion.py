from random import random


def independentCascade(graph, seedNode, probability):
    informed = []
    new_informed = [seedNode]

    for u in new_informed:

        # Remove u from newInformed to informed
        new_informed.remove(u)
        informed.append(u)

        # Iterate over all neighbours of u
        neighbors = graph.neighbors(u)
        for v in neighbors:
            if (v not in new_informed) and (v not in informed) and random() <= probability:
                new_informed.append(v)

    return len(informed)


def weightedCascade(graph, seed_node):
    informed = []
    new_informed = [seed_node]

    for u in new_informed:

        # Remove u from newInformed to informed
        new_informed.remove(u)
        informed.append(u)

        neighbors = graph.neighbors(u)
        for v in neighbors:
            v_d = graph.degree[v]
            if (v not in new_informed) and (v not in informed) and random() <= 1 / v_d:
                new_informed.append(v)

    return len(informed)


def applySpread(graph_seed_pairs, fun, prob=-1):
    result = {}

    # If WC
    if prob == -1:
        for g, u in graph_seed_pairs:
            result[(g, u)] = fun(g, u)
    else:
        for g, u in graph_seed_pairs:
            result[(g, u)] = fun(g, u, prob)

    return result
