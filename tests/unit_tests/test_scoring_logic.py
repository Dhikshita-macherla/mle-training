import unittest

import numpy as np
from sklearn.metrics import mean_squared_error

from housePricePrediction import scoring_logic


class TestModelScoring(unittest.TestCase):
    def test_scoring_logic(self):
        self.y = np.array([2, 3, 4, 5])
        self.y_pred = np.array([2.5, 4, 5, 4.5])
        rmse_exp = np.sqrt(mean_squared_error(self.y, self.y_pred))
        rmse_act, _ = scoring_logic.scoring_logic(self.y, self.y_pred)
        self.assertIsInstance(rmse_act, float)
        self.assertAlmostEqual(rmse_act, rmse_exp, places=4)


if __name__ == '__main__':
    unittest.main()
