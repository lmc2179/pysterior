import random
from math import log
import numpy as np
import pyximport; pyximport.install()
from norm_pdf import lognormpdf

def fixed_variance_gaussian_proposal(mu):
        PROPOSAL_VARIANCE = 0.1
        return np.array([random.gauss(w_i, PROPOSAL_VARIANCE) for w_i in mu])

class NoisyRegressorDistribution(object):
    def get_likelihood(self, w, true_t, expected_t):
        FIXED_OBSERVATION_NOISE_VARIANCE = 1.0
        return self._flat_gaussian_prior(w)+self._observation_likelihood(FIXED_OBSERVATION_NOISE_VARIANCE, expected_t, true_t)

    def _flat_gaussian_prior(self, w):
        FIXED_PRECISION = 0.0001
        probabilities = (lognormpdf(w_i, 0, 1.0/FIXED_PRECISION) for w_i in w)
        return sum(probabilities)

    def _observation_likelihood(self, FIXED_OBSERVATION_NOISE_VARIANCE, expected_t, true_t):
        return lognormpdf(expected_t, true_t, FIXED_OBSERVATION_NOISE_VARIANCE)

class AbstractMetropolisSampler(object):
    LOG_ONE = log(1.0)

    def propose(self, current):
        raise NotImplementedError

    def accept(self, prob_old, prob_new):
        acceptance_prob = min(self.LOG_ONE, prob_new - prob_old)
        if log(random.random()) < acceptance_prob:
            return True
        else:
            return False

    def parameter_likelihood(self, w, true_t, expected_t):
        raise NotImplementedError

class MetropolisRegressionSampler(AbstractMetropolisSampler):
    def __init__(self, regression_function):
        self.regression_function = regression_function

    def propose(self, current):
        return fixed_variance_gaussian_proposal(current)

    def parameter_likelihood(self, w, true_t, expected_t): #TODO: Optimize this, it's ~90% of our compute time
        return NoisyRegressorDistribution().get_likelihood(w, true_t, expected_t)

    def _get_next_parameter_sample(self, w, x, t):
        potential_w = self.propose(w)
        expected_t = self.regression_function(x, w)
        expected_t_potential = self.regression_function(x, potential_w)
        prob_w = self.parameter_likelihood(w, t, expected_t)
        prob_potential_w = self.parameter_likelihood(potential_w, t, expected_t_potential)
        if self.accept(prob_w, prob_potential_w):
            w = potential_w
        return w

    def get_posterior_parameter_samples(self, X, y, iterations, burn_in=None, thinning=None):
        dimension = len(X[0])
        w = np.zeros(dimension)
        samples = []
        for i in range(iterations):
            for x,t in zip(X,y):
                w = self._get_next_parameter_sample(w, x, t)
                if burn_in and i > burn_in and thinning and i%thinning == 0:
                    samples.append(w)
        return samples

class BayesianLinearRegression(object):
    def __init__(self):
        self.sampler = MetropolisRegressionSampler(linear_regression_function)

    def fit_sample(self, X, y, iterations, burn_in=None, thinning=None):
        self.samples = self.sampler.get_posterior_parameter_samples(X, y, iterations, burn_in, thinning)

    def get_parameter_samples(self):
        return np.array(list(zip(*self.samples)))

    def get_parameter_sample_mean(self):
        return np.array([np.sum(p)/len(p) for p in self.get_parameter_samples()])

def linear_regression_function(x,w):
    return x.dot(w)

