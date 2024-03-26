import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def scoring_logic(labels, predictions):
    mse= mean_squared_error(labels, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(labels, predictions)
    return rmse, mae
