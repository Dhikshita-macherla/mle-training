import unittest

import numpy as np
import pandas as pd

from housePricePrediction import data_training


class TestModelTraining(unittest.TestCase):
    def test_stratifiedShuffleSplit(self):
        self.data = {
            'longitude': [-122.23, -122.22, -122.24, -122.25, -122.25],
            'latitude': [37.88, 37.86, 37.85, 37.85, 37.85],
            'housing_median_age': [41, 21, 52, 52, 52],
            'total_rooms': [880, 7099, 1467, 1274, 1627],
            'total_bedrooms': [129, 1106, 190, 235, 280],
            'population': [322, 2401, 496, 558, 565],
            'households': [126, 1138, 177, 219, 259],
            'median_income': [8.3252, 8.3014, 7.2574, 5.6431, 3.8462],
            'ocean_proximity': ['NEAR BAY',
                                'NEAR BAY', 'NEAR BAY', 'NEAR BAY', 'NEAR BAY'],
            'income_cat': [3, 2, 4, 5, 2]
            }

        self.data = pd.DataFrame(self.data)
        train_set, test_set, strain, stest = \
            data_training.stratified_Shuffle_Split(self.data)
        self.assertGreaterEqual(len(set(strain)), 1)
        self.assertEqual(train_set.shape[0], 16512)
        self.assertEqual(test_set.shape[0], 4128)

    def test_train_data_regression(self):
        self.X = np.array([[1], [2], [3]])
        self.y = np.array([1, 2, 3])
        pred = data_training.train_data_regression('lin', self.X, self.y)
        self.assertAlmostEqual(pred[0], 1)


if __name__ == '__main__':
    unittest.main()
    unittest.main()
