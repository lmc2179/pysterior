# This is a pilot of encapsulating pymc3 models
import pymc3
import numpy as np
from matplotlib import pyplot as plt
import unittest

# No. 1: Linear Regression
class AbstractLinearRegression(object):
    def fit(self, X, y, sampling_iterations):
        X = self._force_shape(X)
        model = self._build_model(X, y)
        with model:
            self.map_estimate = pymc3.find_MAP(model=model)
            step = pymc3.NUTS(scaling=self.map_estimate)
            trace = pymc3.sample(sampling_iterations, step, start=self.map_estimate)
        self.samples = trace

    def get_predictive_posterior_samples(self, x):
        "Obtain a sample of the output variable's distribution by running the sample variable values through the model."
        predictive_posterior_samples = []
        for alpha, beta in zip(self.samples['alpha'], self.samples['beta']):
            predictive_posterior_samples.append(alpha + np.dot(x, beta))
        return predictive_posterior_samples

    def predict(self, x):
        "Approximates the expected value of the output variable."
        s = self.get_predictive_posterior_samples(x)
        return sum(s) / len(s)

    def _build_model(self, X, y):
        raise NotImplementedError

    def _force_shape(self, X):
        shape = np.shape(X)
        if len(shape) == 1:
            return np.reshape(X, (shape[0], 1))
        return X

    def get_map_estimate(self):
        return self.map_estimate

    def get_samples(self):
        return self.samples

class LinearRegression(AbstractLinearRegression):
    def _build_model(self, X, y):
        lr_model = pymc3.Model()

        data_length = len(X[0])

        with lr_model:
            alpha_precision = pymc3.Uniform(name='alpha_precision')
            alpha = pymc3.Normal(name='alpha', mu=0, sd=1.0/alpha_precision)
            precision = pymc3.Uniform(name='precision')
            beta = pymc3.Normal(name='beta', mu=0, sd=1.0/precision, shape=data_length)
            sigma = pymc3.HalfNormal(name='sigma', sd=1)
            X = pymc3.Normal(name='X', mu=1, sd=2, observed=X)
            mu = alpha + beta.dot(X.T)
            Y_obs = pymc3.Normal(name='Y_obs', mu=mu, sd=sigma, observed=y)

        return lr_model

class RidgeRegression(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def _build_model(self, X, y):
        lr_model = pymc3.Model()

        data_length = len(X[0])

        with lr_model:
            alpha = pymc3.Normal(name='alpha', mu=0, sd=self.alpha)
            beta = pymc3.Normal(name='beta', mu=0, sd=self.alpha, shape=data_length)
            sigma = pymc3.HalfNormal(name='sigma', sd=1)
            X = pymc3.Normal(name='X', mu=1, sd=2, observed=X)
            mu = alpha + beta.dot(X.T)
            Y_obs = pymc3.Normal(name='Y_obs', mu=mu, sd=sigma, observed=y)

        return lr_model

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

        lr = LinearRegression()
        lr.fit(X, y, 10000)
        samples = lr.get_samples()
        map_estimate = lr.get_map_estimate()
        expected_map = {'alpha': np.array(1.014043926179071), 'beta': np.array([ 1.46737108,  0.29347422]), 'sigma_log': np.array(0.11928775836956886)}
        self.assertAlmostEqual(float(map_estimate['alpha']), float(expected_map['alpha']), delta=1e-1)
        for true_beta, map_beta in zip(map_estimate['beta'], expected_map['beta']):
            self.assertAlmostEqual(true_beta, map_beta, delta=1e-1)
        test_point = X[7]
        true_y = TRUE_ALPHA + TRUE_BETA[0]*test_point[0] + TRUE_BETA[1]*test_point[1]
        print(true_y)
        predicted_y = lr.predict(test_point)
        print(predicted_y)
        self.assertAlmostEqual(true_y, predicted_y, delta=1e-1)

    def test_simple_linear_regression(self):
        np.random.seed(123)
        TRUE_ALPHA, TRUE_SIGMA = 1, 1
        TRUE_BETA = 2.5
        size = 100
        X = np.linspace(0, 1, size)
        noise = (np.random.randn(size)*TRUE_SIGMA)
        y = (TRUE_ALPHA + TRUE_BETA*X + noise)

        lr = LinearRegression()
        lr.fit(X, y, 2000)
        test_point = X[7]
        true_y = TRUE_ALPHA + TRUE_BETA*test_point
        print(true_y)
        predicted_y = lr.predict(test_point)
        print(predicted_y)
        self.assertAlmostEqual(true_y, predicted_y, delta=1e-1)
        # predicted_line = [lr.predict(x) for x in X]
        # plt.plot(X, y, linewidth=0.0, marker='x', color='g')
        # plt.plot(X, predicted_line)
        # plt.show()


if __name__ == '__main__':
    unittest.main()


