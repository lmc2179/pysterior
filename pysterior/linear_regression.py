import random
from functools import reduce
from math import log, exp, pi

import numpy as np
from scipy.stats.distributions import norm

MIN_LOG_PRECISION =1e-315

def lognormpdf(x, mean, sd):
    var = float(sd)**2
    denom = (2*pi*var)**.5
    num = exp(-(float(x)-float(mean))**2/(2*var))
    try:
        return log(num/denom)
    except ValueError:
        return norm.logpdf(x, mean, sd) #TODO: Why does this work when the naive approach does not?

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
        return reduce(lambda x,y:x+y, probabilities, 1.0)

    def _observation_likelihood(self, FIXED_OBSERVATION_NOISE_VARIANCE, expected_t, true_t):
        return lognormpdf(expected_t, true_t, FIXED_OBSERVATION_NOISE_VARIANCE)

class MarkovChainSampler(object):
    LOG_ONE = log(1.0)

    def propose(self, current):
        return fixed_variance_gaussian_proposal(current)

    def accept(self, prob_old, prob_new):
        acceptance_prob = min(self.LOG_ONE, prob_new - prob_old)
        if log(random.random()) < acceptance_prob:
            return True
        else:
            return False

    def parameter_likelihood(self, w, true_t, expected_t): #TODO: Optimize this, it's ~90% of our compute time
        return NoisyRegressorDistribution().get_likelihood(w, true_t, expected_t)

class BayesianLinearRegression(object):
    def __init__(self):
        self.sampler = MarkovChainSampler()
        self.model = LinearRegressionModel()

    def _get_next_parameter_sample(self, w, x, t):
        potential_w = self.sampler.propose(w)
        expected_t = self.model.f(x, w)
        expected_t_potential = self.model.f(x, potential_w)
        prob_w = self.sampler.parameter_likelihood(w, t, expected_t)
        prob_potential_w = self.sampler.parameter_likelihood(potential_w, t, expected_t_potential)
        if self.sampler.accept(prob_w, prob_potential_w):
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
        parameter_samples = zip(*samples)
        for p in parameter_samples:
            print(min(p),max(p))
            print(np.sum(p)/len(p))
            print(np.histogram(p))
        return samples

class LinearRegressionModel(object):
    @staticmethod
    def f(x,w):
        return x.dot(w)

