from math import log
import random

class MetropolisHastings(object):
    """
    A Metropolis-Hastings sampler. Samples from an arbitrary distribution using a proposal distribution.

    The target distribution can be any function, and the proposal distribution must implement the proposal distribution
    interface. (proposal_dist.py/AbstractProposalDistribution).
    """
    LOG_ONE = log(1.0)
    def __init__(self, target_distribution, proposal_distribution):
        self.target_distribution = target_distribution
        self.proposal_distribution = proposal_distribution

    def _propose(self, current_state):
        "Return the next proposed state by delegating to the proposal distribution."
        return self.proposal_distribution.propose(current_state)

    def _evaluate_state(self, state):
        "Return the next proposed state by delegating to the proposal distribution."
        return self.target_distribution(state)

    def _evaluate_proposal(self, current_state, new_state):
        "Evaluate the likelihood of a state transition by delegating to the proposal distribution."
        return self.proposal_distribution.transition_log_probability(current_state, new_state)

    def _accept(self, likelihood_old, likelihood_new):
        "Decide whether or not to accept the new state by applying the Metropolis-Hastings condition."
        acceptance_prob = min(self.LOG_ONE, likelihood_new - likelihood_old)
        if log(random.random()) < acceptance_prob:
            return True
        else:
            return False

    def sample(self, iterations, burn_in, thinning, inital_state):
        """
        Sample from the given distribution by repeated sampling from the proposal distribution.

        :param iterations: an integer number of samples to run
        :param burn_in: the number of samples to discard at the beginning of the chain
        :param thinning: only accept states for iterations which are a multiple of this number
        :return samples: a list of samples from the target distribution
        """
        samples = []
        current_state = inital_state
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

class DynamicProposalMetropolisHastings(MetropolisHastings):
    pass

class ParameterPosteriorSample(MetropolisHastings):
    def __init__(self, prior_log_pdf, data_log_likelihood, proposal):
        target_distribution = self._build_target_distribution(prior_log_pdf, data_log_likelihood)
        super(ParameterPosteriorSample, self).__init__(target_distribution, proposal)