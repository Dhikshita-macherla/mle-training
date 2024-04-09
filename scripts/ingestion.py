
import argparse
import logging
import logging.config
import os

import pandas as pd
from config_logger import configure_logger

from housePricePrediction import data_ingestion, data_training


def ingestion(output_folder, logger):
    # fetch
    logger.info('Fetching data')
    raw_data_path = output_folder+'/raw'
    os.makedirs(raw_data_path, exist_ok=True)
    DOWNLOAD_ROOT = (
        "https://raw.githubusercontent.com/ageron/handson-ml/master/")
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
    data_ingestion.fetch_housing_data(HOUSING_URL, raw_data_path)
    logger.info("Data Downloaded n Extracted Successfully")
    # load
    housing = data_ingestion.load_housing_data(raw_data_path)
    logger.info("Data Loaded Successfully")
    # train-test-split
    train_set, test_set, train, test = (
        data_training.stratified_Shuffle_Split(housing)
    )
    logger.info("train- test split done succesfully")
    # preprocessing

    y_train = train["median_house_value"].copy()
    X_test = test.drop("median_house_value", axis=1)
    y_test = test["median_house_value"].copy()

    logger.info("Preprocessing for train data")
    full_pipeline = data_ingestion.pipelinesIngestion(housing)
    X_train = full_pipeline.fit_transform(train)

    logger.info("Preprocessing for test data")
    X_test = full_pipeline.transform(test)

    # saving op
    processed_data_path = output_folder + '/processed'
    os.makedirs(processed_data_path, exist_ok=True)
    X_train = pd.DataFrame(X_train).to_csv(processed_data_path +
                                           '/X_train.csv',
                                           index=False)
    y_train = pd.DataFrame(y_train).to_csv(processed_data_path +
                                           '/y_train.csv',
                                           index=False)
    X_test = pd.DataFrame(X_test).to_csv(processed_data_path +
                                         '/X_test.csv',
                                         index=False)
    y_test = pd.DataFrame(y_test).to_csv(processed_data_path +
                                         '/y_test.csv',
                                         index=False)
    logger.info("Train Test dataset saved Successfully")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_folder", help="Add path to output folder")
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
    ingestion(args.output_folder, logger)


if __name__ == '__main__':
    main()
