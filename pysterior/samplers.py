from math import log
import random
import numpy as np

class MetropolisHastings(object):
    LOG_ONE = log(1.0)
    def __init__(self, target_distribution):
        self.target_distribution = target_distribution

    def _propose(self, current_state):
        raise NotImplementedError

    def _evaluate_state(self, state):
        return self.target_distribution(state)

    def _evaluate_proposal(self, current_state, new_state):
        raise NotImplementedError

    def _get_initial_value(self):
        raise NotImplementedError

    def _accept(self, likelihood_old, likelihood_new):
        acceptance_prob = min(self.LOG_ONE, likelihood_new - likelihood_old)
        if log(random.random()) < acceptance_prob:
            return True
        else:
            return False

    def sample(self, iterations, burn_in, thinning):
        samples = []
        current_state = self._get_initial_value()
        prob_current_state = self._evaluate_state(current_state)
        for i in range(iterations):
            new_state = self._propose(current_state)
            likelihood_new = self._evaluate_state(new_state) + self._evaluate_proposal(new_state, current_state)
            likelihood_current = prob_current_state + self._evaluate_proposal(current_state, new_state)
            if self._accept(likelihood_current, likelihood_new):
                current_state = new_state
                prob_current_state = likelihood_new
                if i >= (burn_in-1) and i % thinning == 0:
                    samples.append(current_state)
        return samples




class Metropolis(MetropolisHastings):
    def _evaluate_proposal(self, current_state, new_state):
        return self.LOG_ONE

class GaussianMetropolis1D(Metropolis):
    def __init__(self, proposal_variance, target_distribution):
        self.proposal_sigma = proposal_variance
        super(GaussianMetropolis1D, self).__init__(target_distribution)

    def _propose(self, current_state):
        return np.random.normal(current_state, self.proposal_sigma)

    def _get_initial_value(self):
        return random.gauss(0.0, 100)