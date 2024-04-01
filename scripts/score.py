import argparse
import logging
import os
import pickle

import pandas as pd
from config_logger import configure_logger

from housePricePrediction import scoring_logic

logger = logging.getLogger(__name__)


def scoring(data_folder, pred_folder, op_folder, logger):
    X = pd.read_csv(data_folder + '/X_test.csv')
    y = pd.read_csv(data_folder + '/y_test.csv')
    logger.info("Test datas extracted successfully")
    os.makedirs(op_folder, exist_ok=True)
    files = os.listdir(pred_folder)
    for file in files:
        if os.path.isfile(pred_folder+'/'+file):
            with open(os.path.join(pred_folder + '/' + file), 'rb') as f:
                pred_model = pickle.load(f)
                final_predictions_test = pred_model.predict(X)
                final_rmse_test, final_mae_test = scoring_logic.scoring_logic(
                    y, final_predictions_test
                )
                print("Scores calculated for ", file)
                with open(op_folder + '/' + file + "_score.txt", 'w') as f:
                    f.write("RMSE : {}\n".format(final_rmse_test))
                    f.write("MAE : {}".format(final_mae_test))
    logger.info("Scores saved Successfully as txt files")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="Add path to ip folder(datasets)")
    parser.add_argument("pred",
                        help="Add path to trained models folder(pickle files)")
    parser.add_argument("op_file", help="Add path to op folder")
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
    scoring(args.data, args.pred, args.op_file, logger)


if __name__ == '__main__':
    main()
