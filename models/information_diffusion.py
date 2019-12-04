from concurrent.futures.thread import ThreadPoolExecutor
from random import random


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
    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for i in range(iterations):
            future = executor.submit(independentCascadeWorker, graph, probability, seedNode)
            results.append(future.result())

    print("Length", len(results))
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
