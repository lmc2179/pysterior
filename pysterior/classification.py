from math import exp, e
import pymc3
from pysterior import regression
import numpy as np
from theano import tensor as T
from theano import map, function

class AbstractLogisticRegression(regression._AbstractModel):
    def get_predictive_posterior_samples(self, x):
        raise NotImplementedError

    def predict(self, x):
        "Approximates the expected value of the output variable."
        s = self.get_predictive_posterior_samples(x)
        return sum(s) / len(s)

class BinaryLogisticRegressionModel(AbstractLogisticRegression):
    def _build_model(self, X, y):
        lr_model = pymc3.Model()
        with lr_model:
            X = pymc3.Normal(name='X', mu=1, sd=2, observed=X)
            precision = pymc3.Uniform(name='precision')
            w = pymc3.Normal(name='w', mu=0, sd=1.0 / precision, shape=self.input_data_dimension)
            bias_precision = pymc3.Uniform(name='bias_precision')
            bias = pymc3.Normal(name='bias', mu=0, sd=1.0 / bias_precision)
            p = 1.0/(1.0 + e**(-w.dot(X.T) - bias))
            y_obs = pymc3.Bernoulli(p=p, name='y_obs', observed=y)

        return lr_model

    def get_predictive_posterior_samples(self, x):
        "Obtain a sample of the output variable's distribution by running the sample variable values through the model."
        predictive_posterior_samples = []
        for bias, w in zip(self.samples['bias'], self.samples['w']):
            predictive_posterior_samples.append(1.0/(1.0 + e**(-bias - np.dot(x, w))))
        return predictive_posterior_samples

class LogisticRegressionModel(AbstractLogisticRegression):
    def _build_model(self, X, y):
        output_dimension = max(y)+1
        lr_model = pymc3.Model()
        with lr_model:
            X = pymc3.Normal(name='X', observed=X)
            precision = 0.00001
            weights = pymc3.Normal(name='weights',
                                   tau=precision,
                                   shape=(output_dimension, self.input_data_dimension),
                                   testval=T.zeros((output_dimension, self.input_data_dimension)))
            bias_precision = 0.00001
            biases = pymc3.Normal(name='biases', mu=0, tau=bias_precision, shape=output_dimension)
            dots = weights.dot(X.T) # No bias!
            p = T.nnet.softmax(dots + biases.T)
            y_obs = pymc3.Categorical(p=p, name='y_obs', observed=y)

        return lr_model, weights, biases

    def fit(self, X, y, sampling_iterations):
        X = self._force_shape(X)
        self.input_data_dimension = len(X[0])
        model, w, b = self._build_model(X, y)
        with model:
            self.map_estimate = pymc3.find_MAP(model=model, vars=[w, b])
            step = pymc3.NUTS(scaling=self.map_estimate)
            trace = pymc3.sample(sampling_iterations, step, start=self.map_estimate)
        self.samples = trace

    def get_predictive_posterior_samples(self, x):
        "Obtain a sample of the output variable's distribution by running the sample variable values through the model."
        predictive_posterior_samples = []
        for bias, w in zip(self.samples['bias'], self.samples['w']):
            predictive_posterior_samples.append(1.0/(1.0 + e**(-bias - np.dot(x, w))))
        return predictive_posterior_samples