import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.tree import DecisionTreeRegressor


def stratifiedShuffleSplit(housing):
    train_set, test_set = \
        train_test_split(housing, test_size=0.2, random_state=42)

    split = \
        StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    return train_set, test_set, strat_train_set, strat_test_set


def regression(model, housing_prepared, housing_labels):
    if model == 'lin':
        reg = LinearRegression()
    elif model == 'tree':
        reg = DecisionTreeRegressor()
    reg.fit(housing_prepared, housing_labels)
    housing_predictions = reg.predict(housing_prepared)
    return housing_predictions


def cross_validation(model, \
                     housing_prepared, housing_labels, param_distribs, param_grid):
    forest_reg = RandomForestRegressor(random_state=42)
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
    search.fit(housing_prepared, housing_labels)
    cvres = search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)
    return search, cvres, housing_prepared


def predict_Best_Estimator(grid_search, strat_test_set, housing_prepared, imputer):
    feature_importances = grid_search.best_estimator_.feature_importances_
    sorted(zip(feature_importances, housing_prepared.columns), reverse=True)
    final_model = grid_search.best_estimator_

    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    X_test_num = X_test.drop("ocean_proximity", axis=1)
    X_test_prepared = imputer.transform(X_test_num)
    X_test_prepared = pd.DataFrame(
        X_test_prepared, columns=X_test_num.columns, index=X_test.index
    )
    X_test_prepared["rooms_per_household"] = (
        X_test_prepared["total_rooms"] / X_test_prepared["households"]
    )
    X_test_prepared["bedrooms_per_room"] = (
        X_test_prepared["total_bedrooms"] / X_test_prepared["total_rooms"]
    )
    X_test_prepared["population_per_household"] = (
        X_test_prepared["population"] / X_test_prepared["households"]
    )

    X_test_cat = X_test[["ocean_proximity"]]

    X_test_prepared = X_test_prepared.join(
        pd.get_dummies(X_test_cat, drop_first=True)
    )

    final_predictions = final_model.predict(X_test_prepared)
    return y_test, final_predictions
