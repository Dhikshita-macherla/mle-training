import unittest

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from housePricePrediction import data_ingestion, data_training


class TestModelTraining(unittest.TestCase):

    def test_train_data_regression(self):
        self.X = np.array([[1], [2], [3]])
        self.y = np.array([1, 2, 3])
        pred = data_training.train_data_regression('lin', self.X, self.y)
        self.assertAlmostEqual(pred[0], 1)

    def test_stratifiedShuffleSplit(self):
        self.data = data_ingestion.load_housing_data()
        train_set, test_set, strain, stest = \
            data_training.stratified_Shuffle_Split(self.data)
        self.assertGreaterEqual(len(set(strain)), 1)
        self.assertEqual(train_set.shape[0], 16512)
        self.assertEqual(test_set.shape[0], 4128)

    def test_cross_validation(self):
        self.data = data_ingestion.load_housing_data()
        train_set, test_set, strain, stest = \
            data_training.stratified_Shuffle_Split(self.data)
        model= data_training.cross_validation('GridSearchCV', train_set, test_set)
        self.assertIsInstance(model, RandomForestRegressor, "nope")


if __name__ == '__main__':
    unittest.main()
