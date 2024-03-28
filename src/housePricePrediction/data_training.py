import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.tree import DecisionTreeRegressor


def stratifiedShuffleSplit(housing):
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    return train_set, test_set, strat_train_set, strat_test_set


def train_data_regression(model, X, y):
    if model == "lin":
        reg = LinearRegression()
    elif model == "tree":
        reg = DecisionTreeRegressor()
    reg.fit(X, y)
    pred = reg.predict(X)
    return pred


def cross_validation(X, y):
    forest_reg = RandomForestRegressor(random_state=42)
    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
    ]
    search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    search.fit(X, y)
    cvres = search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

    feature_importances = search.best_estimator_.feature_importances_
    sorted(zip(feature_importances, X.columns), reverse=True)
    final_model = search.best_estimator_
    return final_model
