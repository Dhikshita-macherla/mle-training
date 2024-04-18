import argparse
import logging
import logging.config

# import os
import sys

import ingestion
import mlflow
import score
import train
from config_logger import configure_logger


def main(dataset_folder, processed_data_folder, model_output_folder,
         score_output_folder, logger):
    mlflow.set_tracking_uri("http://localhost:2000")
    # mlflow.set_tracking_uri(remote_server_uri)
    print(mlflow.tracking.get_tracking_uri())
    exp_name = "housing"
    mlflow.set_experiment(exp_name)

    # if mlflow.active_run():
    # mlflow.end_run()

    with mlflow.start_run(nested=True, run_name="Main_Run"):
        print('artifact uri:', mlflow.get_artifact_uri())
        mlflow.log_param("dataset_folder", dataset_folder)
        mlflow.log_param("processed_data_folder", processed_data_folder)
        mlflow.log_param("model_output_folder", model_output_folder)
        mlflow.log_param("score_output_folder", score_output_folder)

        try:
            with mlflow.start_run(nested=True, run_name="Ingestion"):
                mlflow.log_param("output_folder", dataset_folder)
                ingestion.ingestion(dataset_folder, logger)
                mlflow.log_artifact(dataset_folder)
                print("Save to: {}".format(mlflow.get_artifact_uri()))

            with mlflow.start_run(nested=True, run_name="training"):
                mlflow.log_param("processed_data_folder",
                                 processed_data_folder)
                mlflow.log_param("model_output_folder",
                                 model_output_folder)
                train.training(processed_data_folder, model_output_folder,
                               logger)

            with mlflow.start_run(nested=True, run_name="scoring"):
                mlflow.log_param("processed_data_folder",
                                 processed_data_folder)
                mlflow.log_param("model_output_folder", model_output_folder)
                mlflow.log_param("score_output_folder", score_output_folder)
                score.scoring(processed_data_folder, model_output_folder,
                              score_output_folder, logger)
                mlflow.log_artifact(score_output_folder)
                print("Save to: {}".format(mlflow.get_artifact_uri()))

        except Exception as e:
            mlflow.log_param("exception", str(e))
            sys.exit(1)


if __name__ == "__main__":
    parser = (argparse.ArgumentParser(description="Script to run\
                                       data ingestion,model training,\
                                      and model scoring."))
    parser.add_argument("dataset_folder",
                        help="Path to the folder containing the dataset.")
    parser.add_argument("processed_data_folder",
                        help="Path to the folder containing processed data")
    parser.add_argument("model_output_folder",
                        help="Path to the output folder \
                            where the trained model will be saved.")
    parser.add_argument("score_output_folder",
                        help="Path to the output folder where \
                            the dataset will be saved.")

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
    main(args.dataset_folder, args.processed_data_folder,
         args.model_output_folder, args.score_output_folder, logger)
