from linear_regression import BayesianLinearRegression
import random
import numpy as np
import unittest

class RegressionTest(unittest.TestCase):
    def test_linear_regression(self):
        TRUE_WEIGHTS = np.array([-13.5,50.0])
        X = np.array([[random.randint(-100,100), random.randint(-100,100)] for i in range(20)])
        y = np.array([TRUE_WEIGHTS.dot(x) for x in X])
        samples = BayesianLinearRegression().fit_sample(X,y, 50000, burn_in=25000, thinning=2)