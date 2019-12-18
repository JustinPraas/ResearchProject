def mainSmall(features, spread_prob, iterations, N, do_knn=False, k=5):
    # Generate small graphs of size N from graph files
    graphs = generateSmallGraphs(N)

    # Machine learning
    result_dict = mainCompute(graphs, features, None, spread_prob, iterations, do_knn, k)

    # Plot
    result_dict['small'] = True
    result_dict['N'] = N
    result_dict['do_knn'] = do_knn
    result_dict['k'] = k
    result_dict['conf_interval'] = mean_confidence_interval(result_dict['y'])
    scatterPlotXDegreeSpread(result_dict)

    print(result_dict['conf_interval'])
    return result_dict


def mainLarge(features, spread_prob, iterations, M, N, do_knn=False, k=5):
    # Generate M graphs of size N
    graphs = generateLargeGraphs(M, N)

    # Machine learning
    result_dict = mainCompute(graphs, features, None, spread_prob, iterations, do_knn, k)

    # Plot
    result_dict['small'] = False
    result_dict['N'] = N
    result_dict['M'] = M
    result_dict['do_knn'] = do_knn
    result_dict['k'] = k
    scatterPlotXDegreeSpread(result_dict)

    return result_dict


def mainCompute(graphs, features, centralityDicts, spread_prob, iterations, do_knn=False, k=5):
    result_dict = {
        'spread_prob': spread_prob,
        'iterations': iterations,
        'features': features
    }

    if centralityDicts is None:
        centralityDicts = getCentralityValuesDict(graphs, features)

    # Build data set
    if concurrent:
        X, y = models.data_set.buildDataSetPar(graphs, centralityDicts, spread_prob, iterations)
    else:
        X, y = models.data_set.buildDataSet(graphs, centralityDicts, spread_prob, iterations)

    result_dict['X'] = X
    result_dict['y'] = y

    # Train-test split
    if doML:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        # Scale
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        if not do_knn:
            # print("Fitting RF using k=10 cross validation")
            rf_gridCV.fit(X_train, y_train)

            # print("Scoring test set")
            result_dict['RFR_R2_test'] = rf_gridCV.score(X_test, y_test)
            result_dict['RFR_R2_train'] = rf_gridCV.score(X_train, y_train)
        else:
            knn_gridCV.fit(X_train, y_train)

            y_pred = knn_gridCV.predict(X_test)
            result_dict['KNN_R2_test'] = r2_score(y_test, y_pred)

    return result_dict


def plotLC(features, M, N, spread_prob, iterations, steps):
    data = mainLarge(features, spread_prob, iterations, M, N)
    makeLearningCurve(data, features, 10, steps)