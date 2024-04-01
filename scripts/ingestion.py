
import argparse
import logging
import logging.config
import os

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
    housing, y_train, X_train = data_ingestion.imputing_data(train)
    X_train = data_ingestion.feature_extraction(X_train)
    X_train = data_ingestion.creating_dummies(train, X_train)

    housing, y_test, X_test = data_ingestion.imputing_data(test)
    X_test = data_ingestion.feature_extraction(X_test)
    X_test = data_ingestion.creating_dummies(housing, X_test)
    logger.info("Preprocessing done Successfully")

    # saving op
    processed_data_path = output_folder + '/processed'
    os.makedirs(processed_data_path, exist_ok=True)
    X_train = X_train.to_csv(processed_data_path + '/X_train.csv', index=False)
    y_train = y_train.to_csv(processed_data_path + '/y_train.csv', index=False)
    X_test = X_test.to_csv(processed_data_path+'/X_test.csv', index=False)
    y_test = y_test.to_csv(processed_data_path+'/y_test.csv', index=False)
    logger.info("Train Test dataset split Successfully")


def main():
    parser = argparse.ArgumentParser()
    print("main")
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
