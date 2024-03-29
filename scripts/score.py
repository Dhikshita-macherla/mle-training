import argparse
import os
import pickle

import pandas as pd

from housePricePrediction import scoring_logic


def scoring(data_folder, pred_folder, op_folder):
    X = pd.read_csv(data_folder + '/X_test.csv')
    y = pd.read_csv(data_folder + '/y_test.csv')
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
                print(final_rmse_test)
                with open(op_folder + '/' + file + "_score.txt", 'w') as f:
                    f.write("RMSE : {}\n".format(final_rmse_test))
                    f.write("MAE : {}".format(final_mae_test))
    print("Scores saved Successfully")


parser = argparse.ArgumentParser()
parser.add_argument("data", help="Add path to ip folder(datasets)")
parser.add_argument("pred", help="Add path to op folder(pickle files)")
parser.add_argument("op_file", help="Add path to op folder(pickle files)")
args = parser.parse_args()
scoring(args.data, args.pred, args.op_file)