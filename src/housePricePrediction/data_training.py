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
    """stratified_Shuffle_Split function
    Performs stratified shuffle split on the given Dataframe

    Parameters
    ----------
    housing : DataFrame
        DataFrame that needs to be split

    Returns
    -------
    train_set : DataFrame
        Train data from train_test_split
    test_set : DataFrame
        Test data from train_test_split
    strat_train_set : DataFrame
        Stratified train dataset
    strat_test_set : DataFrame
        Stratified test dataset

    """
    logger.info("train_test_split Started")
    train_set, test_set = train_test_split(housing,
                                           test_size=0.2,
                                           random_state=42)
    logger.info("Stratified Shuffle Split Started")
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    return train_set, test_set, strat_train_set, strat_test_set


def train_data_regression(model, X, y):
    """train_data_regression function
    Performs appropriate regression based on the model given and
    predicts the values.

    Parameters
    ----------
    model : str
        The type of regression model to use. 'lin' for Linear Regression and
        'tree' for Decision Tree.
    X : DataFrame
        Features for regression, i.e.,
        transformed imputed dataFrame without median_house_value column
    y : array
        Target variable for regression, i.e.,
        DataFrame with only median_house_value column

    Returns
    -------
    pred : array
        Predicted values of X
    reg : Object
        Trained regression model

    """
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
    """cross_validation function
    Performs appropriate cross validation based on the model given and
    finds the best model from all the possibilities.

    Parameters
    ----------
    model : str
        The type of cross validation model to use.
        'RandomizedSearchCV' for Randomized Search cross validation and
        'GridSearchCV' for Grid Search cross validation.
    X : DataFrame
        Features for cross validation, i.e.,
            transformed imputed dataFrame without median_house_value column
    y : array
        Target variable for cross validation, i.e.,
            DataFrame with only median_house_value column

    Returns
    -------
    final_model : Object
        Best estimator after all the possibilities in cross validation

    """
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
    return final_model
