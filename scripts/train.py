import argparse
import os
import pickle

import pandas as pd

from housePricePrediction import data_training


def training(ip_path, op_path):
    housing_X = pd.read_csv(ip_path+'/X_train.csv')
    housing_y = pd.read_csv(ip_path+'/y_train.csv').values.ravel()
    os.makedirs(op_path, exist_ok=True)

    print("Training the model")
    _, lin_model = data_training.train_data_regression("lin", housing_X,
                                                       housing_y)
    with open(op_path + "/linReg_model.pkl", 'wb') as f:
        pickle.dump(lin_model, f)
    _, dtree_model = data_training.train_data_regression("tree", housing_X,
                                                         housing_y)
    with open(op_path+"/deciTree_model.pkl", 'wb') as f:
        pickle.dump(dtree_model, f)
    final_model_rand = data_training.cross_validation('RandomizedSearchCV',
                                                      housing_X,
                                                      housing_y)
    with open(op_path + "/randCV_model.pkl", 'wb') as f:
        pickle.dump(final_model_rand, f)
    # print("Best Estimator for RandomizedSearchCV: ", final_model_rand)
    final_model_grid = data_training.cross_validation('GridSearchCV',
                                                      housing_X,
                                                      housing_y)
    with open(op_path+"/gsCV_model.pkl", 'wb') as f:
        pickle.dump(final_model_grid, f)
    # print("Best Estimator for GridSearchCV: ", final_model_grid)
    print("Training done Successflly")


parser = argparse.ArgumentParser()
parser.add_argument("ip_folder", help="Add path to ip folder(datasets)")
parser.add_argument("op_folder", help="Add path to op folder(pickle files)")
args = parser.parse_args()
training(args.ip_folder, args.op_folder)
