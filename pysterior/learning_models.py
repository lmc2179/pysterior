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
        self.samples = self.sampler.sample(iterations, burn_in, thinning)

    def get_samples(self):
        return self.samples

    def predict_point_estimate(self, X):
        expected_parameter_value = self.samples.get_mean()
        return np.array([self.regression_function(expected_parameter_value, x) for x in X])

    def sample_predictive_posterior(self, x, no_samples=None):
        if not no_samples:
            return np.array([self.regression_function(s, x) for s in self.samples.get_samples()])
        else:
            parameter_samples = self.samples.get_random(no_samples)
            return np.array([self.regression_function(s, x) for s in parameter_samples])