import unittest

import numpy as np
import pandas as pd

from housePricePrediction import data_training


class TestModelTraining(unittest.TestCase):

    def test_train_data_regression(self):
        self.X = np.array([[1], [2], [3]])
        self.y = np.array([1, 2, 3])
        pred = data_training.train_data_regression('lin', self.X, self.y)
        self.assertAlmostEqual(pred[0], 1)





if __name__ == '__main__':
    unittest.main()
