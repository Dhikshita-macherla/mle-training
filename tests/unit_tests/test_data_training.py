import unittest

import numpy as np
import pandas as pd

from housePricePrediction import data_training


class TestModelTraining(unittest.TestCase):
    def test_stratifiedShuffleSplit(self):
        self.data(pd.DataFrame({
            'income_cat': [1, 2, 3, 4, 3, 1],
            'total_rooms': [50, 100, 60, 70, 200, 150],
            'households': [60, 10, 20, 40, 50, 35]}))
        train_set, test_set, _, _ = \
            data_training.stratifiedShuffleSplit(self.data)
        self.assertEqual(train_set.shape[0], 4)
        self.assertEqual(test_set.shape[0], 2)

    def test_train_data_regression(self):
        self.X = np.array([[1], [2], [3]])
        self.y = np.array([1, 2, 3])
        pred = data_training.train_data_regression('lin', self.X, self.y)
        self.assertAlmostEqual(pred[0], 1)


if __name__ == '__main__':
    unittest.main()
