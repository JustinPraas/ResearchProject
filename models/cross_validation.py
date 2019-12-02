import numpy as np

# Number of trees in random forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

n_estimators = [int(x) for x in np.linspace(start=20, stop=100, num=10)]
# n_estimators = 50

# Number of features to consider at every split
# max_features = ['auto', 'sqrt']

# Maximum number of levels in tree #!!! WE Don't need this
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [0.001, 0.01, 0.1]

# Minimum number of samples required at each leaf node
min_samples_leaf = [0.001, 0.01, 0.1]

# Method of selecting samples for training each tree
# bootstrap = [True, False]

param_grid = {'n_estimators': n_estimators,
               # 'max_features': max_features,
              # 'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf
              }

regressor = RandomForestRegressor()
rf_gridCV = GridSearchCV(estimator=regressor,
                         param_grid=param_grid,
                         cv=5,
                         scoring="r2",
                         verbose=1,
                         n_jobs=-1)
