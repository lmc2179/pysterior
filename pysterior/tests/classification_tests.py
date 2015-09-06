import unittest
from pysterior import classification
import numpy as np

class LogisticRegressionTest(unittest.TestCase):
    def test_binary_log_reg(self):
        X = [[1, 2], [0, 1], [1, 0], [2, 0]]
        y = [1, 1,  0, 0]
        lr = classification.BinaryLogisticRegressionModel()
        lr.fit(X, y, 1000)
        for x in X:
            print(lr.get_predictive_posterior_samples(x))
            print(lr.predict(x))

    def test_log_reg(self):
        X = np.array([[1, 1], [-1, 1], [-1, 1], [-1, -1]])
        y = [0, 1,  0, 1]
        lr = classification.LogisticRegressionModel()
        lr.fit(X, y, 1000)
        for x in X:
            print(lr.get_predictive_posterior_samples(x))
            print(lr.predict(x))