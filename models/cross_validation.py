import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor

n_estimators = [int(x) for x in np.linspace(start=20, stop=100, num=10)]

# Minimum number of samples required to split a node
min_samples_split = [0.001, 0.01, 0.1]

# Minimum number of samples required at each leaf node
min_samples_leaf = [0.001, 0.01, 0.1]

param_grid = {'n_estimators': n_estimators,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf
              }

regressor = RandomForestRegressor()
rf_gridCV = GridSearchCV(estimator=regressor,
                         param_grid=param_grid,
                         cv=10,
                         scoring="r2",
                         verbose=1,
                         n_jobs=-1)


''' KNN '''
knn = KNeighborsRegressor()
param_grid = {'n_neighbors': np.arange(2, 25)}
knn_gridCV = GridSearchCV(knn, param_grid, cv=5)