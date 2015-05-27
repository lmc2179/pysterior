import numpy as np
import random
from functools import reduce
from scipy.stats.distributions import norm
from math import log, exp

LOG_ONE = log(1.0)

def lognormpdf(x, mean, sd):
    var = float(sd)**2
    pi = 3.1415926
    denom = (2*pi*var)**.5
    num = exp(-(float(x)-float(mean))**2/(2*var))
    return log(num/denom)

class MarkovChainSampler(object):
    @staticmethod
    def propose(current):
        PROPOSAL_VARIANCE = 0.1
        return np.array([random.gauss(w_i, PROPOSAL_VARIANCE) for w_i in current])

    @staticmethod
    def accept(prob_old, prob_new):
        acceptance_prob = min(LOG_ONE, prob_new - prob_old)
        if log(random.random()) < acceptance_prob:
            return True
        else:
            return False

    @classmethod
    def parameter_likelihood(cls, w, true_t, expected_t): #TODO: Optimize this, it's ~90% of our compute time
        FIXED_OBSERVATION_NOISE_VARIANCE = 1.0
        return cls._prior(w)+cls._observation_likelihood(FIXED_OBSERVATION_NOISE_VARIANCE, expected_t, true_t)

    @staticmethod
    def _prior(w):
        FIXED_PRECISION = 0.0001
        probabilities = (lognormpdf(w_i, 0, 1.0/FIXED_PRECISION) for w_i in w)
        return reduce(lambda x,y:x+y, probabilities, 1.0)

    @classmethod
    def _observation_likelihood(cls, FIXED_OBSERVATION_NOISE_VARIANCE, expected_t, true_t):
        return norm.logpdf(expected_t, true_t, FIXED_OBSERVATION_NOISE_VARIANCE)

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

import cProfile
X = np.array([[0,0],[0,1], [1,1], [50,2]])
y = np.array([0,50.5,50.1, 101])
cProfile.run('BayesianLinearRegression().get_posterior_parameter_samples(X,y, 15000, burn_in=5000, thinning=2)',
             sort='cumtime')