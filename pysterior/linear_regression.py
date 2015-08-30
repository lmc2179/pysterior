import numpy as np
import pymc3


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


class RidgeRegression(AbstractLinearRegression):
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