import numpy as np
import random
from functools import reduce
from scipy.stats.distributions import norm

class MarkovChainSampler(object):
    @staticmethod
    def propose(current):
        PROPOSAL_VARIANCE = 0.1
        return np.array([random.gauss(w_i, PROPOSAL_VARIANCE) for w_i in current])

    @staticmethod
    def prior(w):
        FIXED_PRECISION = 0.0001
        probabilities = norm.pdf(w, 0, 1.0/FIXED_PRECISION)
        return reduce(lambda x,y:x*y, probabilities, 1.0)

    @staticmethod
    def accept(prob_old, prob_new):
        acceptance_prob = min(1, prob_new/prob_old)
        if random.random() < acceptance_prob:
            return True
        else:
            return False

    @classmethod
    def parameter_likelihood(cls, w, true_t, expected_t):
        FIXED_OBSERVATION_NOISE_VARIANCE = 1.0
        return cls.prior(w)*norm.pdf(expected_t, true_t, FIXED_OBSERVATION_NOISE_VARIANCE)

class BayesianLinearRegression(object):
    @staticmethod
    def get_posterior_samples(X, y, iterations, burn_in=None, thinning=None):
        dimension = len(X[0])
        w = np.zeros(dimension)
        samples = []
        sampler = MarkovChainSampler()
        model = LinearRegressionModel()
        for i in range(iterations):
            for x,t in zip(X,y):
                potential_w = sampler.propose(w)
                expected_t = model.f(x,w)
                expected_t_potential = model.f(x, potential_w)
                prob_w = sampler.parameter_likelihood(w, t, expected_t)
                prob_potential_w = sampler.parameter_likelihood(potential_w, t, expected_t_potential)
                if sampler.accept(prob_w, prob_potential_w):
                    w = potential_w
                if burn_in and i > burn_in and thinning and i%thinning == 0:
                    samples.append(w)
        parameter_samples = zip(*samples)
        for p in parameter_samples:
            print(min(p),max(p))
            print(np.sum(p)/len(p))
            print(np.histogram(p))

class LinearRegressionModel(object):
    @staticmethod
    def f(x,w):
        return x.dot(w)

import cProfile
X = np.array([[0,0],[0,1], [1,1], [50,2]])
y = np.array([0,50.5,50.1, 101])
cProfile.run('BayesianLinearRegression.get_posterior_samples(X,y, 15000, burn_in=5000, thinning=2)')