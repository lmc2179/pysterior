# This is a pilot of encapsulating pymc3 models
import unittest
import numpy as np
from regression import LinearRegression, RidgeRegression, RobustLinearRegression, LassoRegression

class LinearRegressionTest(unittest.TestCase):
    def _run_multiple_linear_regression_for_model(self, model):
        np.random.seed(123)
        TRUE_ALPHA, TRUE_SIGMA = 1, 1
        TRUE_BETA = [1, 2.5]
        size = 100
        X1 = np.linspace(0, 1, size)
        X2 = np.linspace(0,.2, size)
        y = TRUE_ALPHA + TRUE_BETA[0]*X1 + TRUE_BETA[1]*X2 + np.random.randn(size)*TRUE_SIGMA
        X = np.array(list(zip(X1, X2)))
        lr = model
        lr.fit(X, y, 2000)
        samples = lr.get_samples()
        map_estimate = lr.get_map_estimate()
        expected_map = {'alpha': np.array(1.014043926179071), 'beta': np.array([ 1.46737108,  0.29347422]), 'sigma_log': np.array(0.11928775836956886)}
        print('Diff between expected map alpha and true alpha: ', float(map_estimate['alpha']) - float(expected_map['alpha']))
        for true_beta, map_beta in zip(map_estimate['beta'], expected_map['beta']):
            print('Diff between expected map beta and true beta: ', true_beta - map_beta)
        test_point = np.array([X[7]])
        true_y = TRUE_ALPHA + TRUE_BETA[0]*test_point[0][0] + TRUE_BETA[1]*test_point[0][1]
        print(true_y)
        predicted_y = lr.predict(test_point)
        print(predicted_y)
        self.assertAlmostEqual(true_y, predicted_y, delta=1e-1)

    def test_multiple_linear_regression(self):
        self._run_multiple_linear_regression_for_model(LinearRegression())

    def test_multiple_linear_regression_ridge(self):
        self._run_multiple_linear_regression_for_model(RidgeRegression(1000))

    def test_multiple_linear_regression_lasso(self):
        self._run_multiple_linear_regression_for_model(LassoRegression(1000))

    def test_multiple_linear_regression_robust(self):
        self._run_multiple_linear_regression_for_model(RobustLinearRegression())

    def _run_simple_linear_regression_for_model(self, model):
        np.random.seed(123)
        TRUE_ALPHA, TRUE_SIGMA = 1, 1
        TRUE_BETA = 2.5
        size = 100
        X = np.linspace(0, 1, size)
        noise = (np.random.randn(size)*TRUE_SIGMA)
        y = (TRUE_ALPHA + TRUE_BETA*X + noise)

        lr = model
        lr.fit(X, y, 20)
        test_point = np.array([X[7]])
        true_y = TRUE_ALPHA + TRUE_BETA*test_point
        print(true_y)
        predicted_y = lr.predict(test_point)[0]
        print(predicted_y)
        self.assertAlmostEqual(true_y, predicted_y, delta=1e-1)

        multiple_test_points = X[7:9]
        true_y = TRUE_ALPHA + TRUE_BETA*multiple_test_points
        print(true_y)
        predicted_y = lr.predict(multiple_test_points)
        print(predicted_y)

    def test_simple_linear_regression(self):
        self._run_simple_linear_regression_for_model(LinearRegression())

    def test_simple_linear_regression_ridge(self):
        self._run_simple_linear_regression_for_model(RidgeRegression(1000))

    def test_simple_linear_regression_lasso(self):
        self._run_simple_linear_regression_for_model(LassoRegression(1000))

    def test_simple_linear_regression_robust(self):
        self._run_simple_linear_regression_for_model(RobustLinearRegression())

if __name__ == '__main__':
    unittest.main()


