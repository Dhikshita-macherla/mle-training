from housePricePrediction import data_ingestion, data_training, scoring_logic

#fetch
data_ingestion.fetch_housing_data()
#load
housing = data_ingestion.load_housing_data()

#train-test split
train_set, test_set, strat_train_set, strat_test_set= data_training.stratifiedShuffleSplit(housing)

#preprocessing
compare_props_train = data_ingestion.preprocessing(housing, strat_train_set, strat_test_set, test_set)

#Data Visualiztion for train set
housing_train = strat_train_set.copy()
print("Data Visualization for train set")
data_ingestion.data_visualization(housing_train)

#Data Visualiztion for test set
housing_test = strat_train_set.copy()
print("Data Visualization for test set")
data_ingestion.data_visualization(housing_test)

#Feature Extraction for train set
housing_train ,y_train, housing_prepared_train =data_ingestion.imputing_data(strat_train_set)
housing_prepared_train= data_ingestion.feature_extraction(housing_prepared_train)
housing_prepared_train= data_ingestion.creating_dummies(housing_train, housing_prepared_train)
#Feature Extraction for test set
housing_test ,y_test, housing_prepared_test =data_ingestion.imputing_data(strat_test_set)
housing_prepared_test= data_ingestion.feature_extraction(housing_prepared_test)
housing_prepared_test= data_ingestion.creating_dummies(housing_test, housing_prepared_test)

#train model for training set
housing_predictions_train = data_training.train_data_regression('lin', housing_prepared_train, y_train)
lin_rmse_train, lin_mae_train = scoring_logic.scoring_logic(y_train, housing_predictions_train)

housing_predictions = data_training.train_data_regression('tree', housing_prepared_train, y_train)
tree_rmse_train, tree_mae_train = scoring_logic.scoring_logic(y_train, housing_predictions_train)

rnd_search, cvres, housing_prepared_train = data_training.cross_validation('RandomizedSearchCV', housing_prepared_train, y_train)
grid_search, cvres, housing_prepared_train = data_training.cross_validation('GridSearchCV', housing_prepared_train, y_train)

final_model_train = data_training.predict_Best_Estimator(grid_search, housing_prepared_train)
print("Best Estimator", final_model_train)

final_predictions_train = final_model_train.predict(housing_prepared_train)
final_rmse_train, final_mae_train = scoring_logic.scoring_logic(y_train, final_predictions_train)
print("Scoring metrics for train-data: \n", final_rmse_train,"   ", final_mae_train)  #scoring for train set

#test using trained models
final_predictions_test = final_model_train.predict(housing_prepared_test)
final_rmse_test, final_mae_test = scoring_logic.scoring_logic(y_test, final_predictions_test)
print("Scoring metrics for test-data: \n", final_rmse_test,"   ", final_mae_test)   #scoring for test set
