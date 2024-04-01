import logging
import logging.config

import numpy as np
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.tree import DecisionTreeRegressor

logger = logging.getLogger(__name__)


def stratified_Shuffle_Split(housing):
    logger.info("train_test_split Started")
    train_set, test_set = train_test_split(housing,
                                           test_size=0.2,
                                           random_state=42)
    logger.info("train_test_split for train and test set done succesfully")
    logger.info("Stratified Shuffle Split Started")
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    logger.info("Stratified Shuffle Split done successfully")
    return train_set, test_set, strat_train_set, strat_test_set


def train_data_regression(model, X, y):
    logger.info("Regression Started")
    if model == "lin":
        reg = LinearRegression()
    elif model == "tree":
        reg = DecisionTreeRegressor()
    reg.fit(X, y)
    pred = reg.predict(X)
    logger.info("model is fitted and y values for X is predicted successfully")
    return pred, reg


def cross_validation(model, X, y):
    logger.info("Cross validation Started")
    forest_reg = RandomForestRegressor(random_state=42)
    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {"bootstrap": [False], "n_estimators": [3, 10],
         "max_features": [2, 3, 4]},
    ]
    param_distribs = {
        "n_estimators": randint(low=1, high=200),
        "max_features": randint(low=1, high=8),
    }
    if model == 'RandomizedSearchCV':
        search = RandomizedSearchCV(
            forest_reg,
            param_distributions=param_distribs,
            n_iter=10,
            cv=5,
            scoring="neg_mean_squared_error",
            random_state=42,
        )
    elif model == 'GridSearchCV':
        search = GridSearchCV(
            forest_reg,
            param_grid,
            cv=5,
            scoring="neg_mean_squared_error",
            return_train_score=True,
        )
    search.fit(X, y)
    logger.info("Model fitted successfully")
    cvres = search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

    feature_importances = search.best_estimator_.feature_importances_
    sorted(zip(feature_importances, X.columns), reverse=True)
    final_model = search.best_estimator_
    logger.info("Best model is found successfully")
    logger.info("Cross validation using ", model, "Started")
    return final_model
    return final_model
