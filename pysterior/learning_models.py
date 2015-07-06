import parametric_functions
import numpy as np

#TODO: Credible interval prediction

class RegressionModel(object):
    def __init__(self, target_energy, sampler_class, regression_function):
        """
        :type target_energy abstract_differentiable_function.AbstractDifferentiableFunction
        :type sampler_class sampler.AbstractHamiltonianSampler.__class__
        :type regression_function parametric_functions.SupervisedRegressionFunction
        """
        self.sampler = sampler_class(target_energy=target_energy)
        self.regression_function = regression_function
        self.samples = None

    def sample_parameter_posterior(self, iterations, burn_in=0, thinning=1):
        """
        Samples from the posterior distribution of the model parameters.

        :param iterations: Number of iterations to sample for.
        :param burn_in: Number of samples to discard from beginning of model as the burn-in period
        :param thinning: Integer, save only every n samples to decrease correlation.
        :return: Nothing
        """
        self.samples = self.sampler.sample(iterations, burn_in, thinning)

    def get_samples(self):
        return self.samples

    def predict_point_estimate(self, X):
        """
        Return a prediction for each input in X.
        The parameter value use to predict is the expected value of the model's parameters after sampling.

        :param X: array-like, a list of input points
        :return: array-like, a list of predicted output points
        """
        expected_parameter_value = self.samples.get_mean()
        return np.array([self.regression_function(expected_parameter_value, x) for x in X])

    def sample_predictive_posterior(self, x, no_samples=None):
        """
        Sample from the predictive posterior for a given input point.

        :param x: An input point.
        :param no_samples: Number of samples of the posterior distribution to take. If default, returns as many samples
            as the model currently has saved.
        :return: array_like, a sequence of samples of the predictive distribution
        """
        if not no_samples:
            return np.array([self.regression_function(s, x) for s in self.samples.get_samples()])
        else:
            parameter_samples = self.samples.get_random(no_samples)
            return np.array([self.regression_function(s, x) for s in parameter_samples])