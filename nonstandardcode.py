from scipy.stats import randint

from housePricePrediction import data_ingestion, data_training, scoring_logic

data_ingestion.fetch_housing_data()
housing = data_ingestion.load_housing_data()

train_set, test_set, strat_train_set, strat_test_set= data_training.stratifiedShuffleSplit(housing)

compare_props = data_ingestion.preprocessing(housing, strat_test_set, test_set)

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing = strat_train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude")
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

corr_matrix = housing.corr(numeric_only=True)
corr_matrix["median_house_value"].sort_values(ascending=False)

housing, housing_labels, housing_prepared, imputer= data_ingestion.feature_extraction(housing, strat_train_set)

housing_predictions = data_training.regression('lin', housing_prepared, housing_labels)
lin_rmse, lin_mae = scoring_logic.scoring_logic(housing_labels, housing_predictions)

housing_predictions = data_training.regression('tree', housing_prepared, housing_labels)
tree_rmse, tree_mae = scoring_logic.scoring_logic(housing_labels, housing_predictions)

param_distribs = {
    "n_estimators": randint(low=1, high=200),
    "max_features": randint(low=1, high=8),
}
param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
    ]
rnd_search, cvres, housing_prepared = data_training.cross_validation('RandomizedSearchCV', housing_prepared, housing_labels, param_distribs, param_grid)
grid_search, cvres, housing_prepared = data_training.cross_validation('GridSearchCV', housing_prepared, housing_labels, param_distribs, param_grid)

feature_importances = grid_search.best_estimator_.feature_importances_
sorted(zip(feature_importances, housing_prepared.columns), reverse=True)
final_model = grid_search.best_estimator_


y_test, final_predictions = data_training.predict_Best_Estimator(grid_search, strat_test_set,housing_prepared, imputer)
final_rmse, final_mae = scoring_logic.scoring_logic(y_test, final_predictions)
print(final_rmse,"   ", final_mae)
