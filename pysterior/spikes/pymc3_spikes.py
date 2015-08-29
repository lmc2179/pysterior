# This is a pilot of encapsulating pymc3 models
import pymc3
import numpy as np
from matplotlib import pyplot as plt
import unittest

# No. 1: Linear Regression
class BayesianLinearRegression(object):
    def sample(self, X, y, iterations):
        X = self._force_shape(X)
        model = self._build_model(X, y)
        with model:
            self.map_estimate = pymc3.find_MAP(model=model)
            step = pymc3.NUTS(scaling=self.map_estimate)
            trace = pymc3.sample(iterations, step, start=self.map_estimate)
        return trace

    def _force_shape(self, X):
        shape = np.shape(X)
        if len(shape) == 1:
            return np.reshape(X, (shape[0], 1))
        return X

    def _build_model(self, X, y):
        lr_model = pymc3.Model()

        data_length = len(X[0])

        with lr_model:
            alpha = pymc3.Normal(name='alpha', mu=0, sd=20)
            precision = pymc3.Uniform(name='precision')
            beta = pymc3.Normal(name='beta', mu=0, sd=1.0/precision, shape=data_length)
            sigma = pymc3.HalfNormal(name='sigma', sd=1)
            X = pymc3.Normal(name='X', mu=1, sd=2, observed=X)
            mu = alpha + beta.dot(X.T)
            Y_obs = pymc3.Normal(name='Y_obs', mu=mu, sd=sigma, observed=y)

        return lr_model

    def get_map_estimate(self):
        return self.map_estimate

class LinearRegressionTest(unittest.TestCase):
    def test_multiple_linear_regression(self):
        np.random.seed(123)
        TRUE_ALPHA, TRUE_SIGMA = 1, 1
        TRUE_BETA = [1, 2.5]
        size = 100
        X1 = np.linspace(0, 1, size)
        X2 = np.linspace(0,.2, size)
        y = TRUE_ALPHA + TRUE_BETA[0]*X1 + TRUE_BETA[1]*X2 + np.random.randn(size)*TRUE_SIGMA

        X = np.array(list(zip(X1, X2)))

        lr = BayesianLinearRegression()
        samples = lr.sample(X, y, 1)
        map_estimate = lr.get_map_estimate()
        expected_map = {'alpha': np.array(1.014043926179071), 'beta': np.array([ 1.46737108,  0.29347422]), 'sigma_log': np.array(0.11928775836956886)}
        self.assertAlmostEqual(float(map_estimate['alpha']), float(expected_map['alpha']), delta=1e-3)
        for true_beta, map_beta in zip(map_estimate['beta'], expected_map['beta']):
            self.assertAlmostEqual(true_beta, map_beta, delta=1e-3)

    def test_simple_linear_regression(self):
        np.random.seed(123)
        TRUE_ALPHA, TRUE_SIGMA = 1, 1
        TRUE_BETA = 2.5
        size = 100
        X = np.linspace(0, 1, size)
        noise = (np.random.randn(size)*TRUE_SIGMA)
        y = (TRUE_ALPHA + TRUE_BETA*X + noise)

        lr = BayesianLinearRegression()
        samples = lr.sample(X, y, 1)
        map_estimate = lr.get_map_estimate()
        print(map_estimate)
        # plt.plot(X, y, linewidth=0.0, marker='x')
        # plt.show()

if __name__ == '__main__':
    unittest.main()


