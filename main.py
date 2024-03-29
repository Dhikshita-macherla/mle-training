from housePricePrediction import data_ingestion, data_training, scoring_logic

# fetch
data_ingestion.fetch_housing_data()
# load
housing = data_ingestion.load_housing_data()

# train-test split
train_set, test_set, strat_train_set, strat_test_set = (
    data_training.stratified_Shuffle_Split(housing)
)

# preprocessing
compare_props = data_ingestion.preprocessing(
    housing, strat_train_set, strat_test_set, test_set
)

# Data Visualiztion for train set
housing_train = strat_train_set.copy()
print("Data Visualization for train set")
data_ingestion.data_visualization(housing_train)

# Data Visualiztion for test set
housing_test = strat_train_set.copy()
print("Data Visualization for test set")
data_ingestion.data_visualization(housing_test)

# Feature Extraction for train set
housing_train, housing_y_train, housing_X_train = data_ingestion.imputing_data(
    strat_train_set
)
housing_X_train = data_ingestion.feature_extraction(housing_X_train)
housing_X_train = data_ingestion.creating_dummies(housing_train,
                                                  housing_X_train)
# Feature Extraction for test set
housing_test, housing_y_test, housing_X_test = data_ingestion.imputing_data(
    strat_test_set
)
housing_X_test = data_ingestion.feature_extraction(housing_X_test)
housing_X_test = data_ingestion.creating_dummies(housing_test, housing_X_test)

# train model for training set
housing_predictions_lin, lin_model = data_training.train_data_regression(
    "lin", housing_X_train, housing_y_train
)


lin_rmse_train, lin_mae_train = scoring_logic.scoring_logic(
    housing_y_train, housing_predictions_lin
)

housing_predictions_reg, dtree_model = data_training.train_data_regression(
    "tree", housing_X_train, housing_y_train
)
tree_rmse_train, tree_mae_train = scoring_logic.scoring_logic(
    housing_y_train, housing_predictions_reg
)
final_model_train_random = data_training.cross_validation('RandomizedSearchCV',
                                                          housing_X_train,
                                                          housing_y_train)
print("Best Estimator", final_model_train_random)
final_model_train_grid = data_training.cross_validation('GridSearchCV',
                                                        housing_X_train,
                                                        housing_y_train)
print("Best Estimator", final_model_train_grid)

final_predictions_train = final_model_train_grid.predict(housing_X_train)
final_rmse_train, final_mae_train = scoring_logic.scoring_logic(
    housing_y_train, final_predictions_train
)
# scoring for train set
print("Scoring for train-data: \n", final_rmse_train, "   ", final_mae_train)


# test using trained models
final_predictions_test = final_model_train_grid.predict(housing_X_test)
final_rmse_test, final_mae_test = scoring_logic.scoring_logic(
    housing_y_test, final_predictions_test
)
# scoring for test set
print("Scoring for test-data: \n", final_rmse_test, "   ", final_mae_test)
