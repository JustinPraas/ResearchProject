from random import random

from joblib import Parallel, delayed


def independentCascade(graph, seedNode, probability, iterations):
    result = []

    for i in range(0, iterations):
        active = []
        target = [seedNode]

        while len(target) > 0:
            u = target.pop()
            active.append(u)
            for v in graph.neighbors(u):
                if v not in active and v not in target:
                    if random() <= probability:
                        target.append(v)
        result.append(len(active))

    average = sum(result) / len(result)
    return average


def independentCascadePar(graph, seedNode, probability, iterations):
    results = Parallel(n_jobs=-1, verbose=0)(delayed(independentCascadeWorker)(graph, probability, seedNode) for i in range(0, iterations))
    average = sum(results) / len(results)
    return average


def independentCascadeWorker(graph, probability, seedNode):
    active = []
    target = [seedNode]
    while len(target) > 0:
        u = target.pop()
        active.append(u)
        for v in graph.neighbors(u):
            if v not in active and v not in target:
                if random() <= probability:
                    target.append(v)
    return len(active)


def weightedCascade(graph, seedNode, iterations):
    result = []

    for i in range(0, iterations):
        active = []
        target = [seedNode]

        while len(target) > 0:
            u = target.pop()
            active.append(u)
            for v in graph.neighbors(u):
                v_d = graph.degree[v]
                if v not in active and v not in target:
                    if random() <= 1 / v_d:
                        target.append(v)
        result.append(len(active))

    average = sum(result) / len(result)
    return average
