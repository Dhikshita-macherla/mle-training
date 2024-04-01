import logging
import logging.config

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


def scoring_logic(y, pred):
    logger.info("Scoring the model started")
    mse = mean_squared_error(y, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, pred)
    logger.info("Calculated the mean_squared_error and mean_absolute_error")
    return rmse, mae
