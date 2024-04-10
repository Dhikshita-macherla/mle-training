# import argparse
# import logging
# import logging.config
import os
import pickle

import mlflow
import pandas as pd

from housePricePrediction import data_training

# from config_logger import configure_logger
# logger = logging.getLogger(__name__)


def training(ip_path, op_path, logger):
    housing_X = pd.read_csv(ip_path+'/X_train.csv')
    housing_y = pd.read_csv(ip_path+'/y_train.csv').values.ravel()
    os.makedirs(op_path, exist_ok=True)
    logger.info("Processed train data extracted successfully")
    logger.info("Training the model with linear regression")
    _, lin_model = data_training.train_data_regression("lin", housing_X,
                                                       housing_y)
    with open(op_path + "/linReg_model.pkl", 'wb') as f:
        pickle.dump(lin_model, f)
        mlflow.sklearn.log_model(lin_model, "lin_model")

    logger.info("Training the model with decision tree")
    _, dtree_model = data_training.train_data_regression("tree", housing_X,
                                                         housing_y)
    with open(op_path+"/deciTree_model.pkl", 'wb') as f:
        pickle.dump(dtree_model, f)
        mlflow.sklearn.log_model(dtree_model, "dtree_model")

    logger.info("Randomized Search CV started")
    final_model_rand = data_training.cross_validation('RandomizedSearchCV',
                                                      housing_X,
                                                      housing_y)
    with open(op_path + "/randCV_model.pkl", 'wb') as f:
        pickle.dump(final_model_rand, f)
        mlflow.sklearn.log_model(final_model_rand, "final_model_rand")

    # print("Best Estimator for RandomizedSearchCV: ", final_model_rand)
    logger.info("Grid search CV started")
    final_model_grid = data_training.cross_validation('GridSearchCV',
                                                      housing_X,
                                                      housing_y)
    with open(op_path+"/gsCV_model.pkl", 'wb') as f:
        pickle.dump(final_model_grid, f)
        mlflow.sklearn.log_model(final_model_grid, "final_model_grid")

    # print("Best Estimator for GridSearchCV: ", final_model_grid)
    logger.info("Training done Successflly \
                and models are stored as pickle files")


'''def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ip_folder", help="Add path to ip folder(datasets)")
    parser.add_argument("op_folder",
                        help="Add path to op folder(pickle files)")
    parser.add_argument("--log-level", help="Specify log level",
                        default="INFO")
    parser.add_argument("--log-path", help="Path to write logs to file")
    parser.add_argument("--no-console-log",
                        action="store_false",
                        dest="console_log",
                        help="Disable writing logs to console",)
    args = parser.parse_args()

    console = args.console_log
    logger = configure_logger(
        log_level=(
            logging.getLevelName(args.log_level.upper())
            if args.log_level
            else logging.DEBUG
        ),
        console=(console),
        log_file=args.log_path,
    )
    training(args.ip_folder, args.op_folder, logger)


if __name__ == '__main__':
    main()'''
