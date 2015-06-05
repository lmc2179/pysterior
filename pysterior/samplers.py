from math import log
import random
import numpy as np
import abc

class AbstractDistribution(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def log_pdf(self, state):
        """The log-likelihood of the target distribution in this state."""

class AbstractParameterPosteriorDistribution(AbstractDistribution): #TODO: Needs factory, unit test
    """Posterior distribution P(w|x) for some distribution over x with parameter w."""
    __metaclass__ = abc.ABCMeta

    def __init__(self, data):
        self.data = data

    @abc.abstractmethod
    def _prior_log_pdf(self, param_value):
        """Evaluation of prior for this value of the parameter."""

    @abc.abstractmethod
    def _observation_log_pdf(self, observations, param_value):
        """Likelihood function for this data set given a particular value for the parameter."""

class AbstractProposalDistribution(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def transition_log_probability(self, current_state, new_state):
        """Return the log likelihood of a transition under this proposal distribution."""

    @abc.abstractmethod
    def propose(self, current_state):
        """Propose a new state given the current state."""

def target_distribution_factory(log_pdf):
    class TargetDistribution(AbstractDistribution):
        def log_pdf(self, x):
            return log_pdf(x)

    return TargetDistribution()

class GaussianProposalDistribution(AbstractProposalDistribution):
    def __init__(self, sigma):
        self.sigma = sigma

    def propose(self, current_state):
        return np.random.normal(current_state, self.sigma)

class MetropolisHastings(object):
    LOG_ONE = log(1.0)
    def __init__(self, target_distribution, proposal_distribution):
        if isinstance(target_distribution, AbstractDistribution):
            self.target_distribution = target_distribution
        else:
            self.target_distribution = target_distribution_factory(target_distribution)
        self.proposal_distribution = proposal_distribution

    def _propose(self, current_state):
        return self.proposal_distribution.propose(current_state)

    def _evaluate_state(self, state):
        return self.target_distribution.log_pdf(state)

    def _evaluate_proposal(self, current_state, new_state):
        return self.proposal_distribution.transition_log_probability(current_state, new_state)

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
        "Return log(1), because the transition probabilities are assumed to cancel since they are symmetric."
        return self.LOG_ONE

class GaussianMetropolis1D(Metropolis):
    def __init__(self, proposal_variance, target_distribution):
        super(GaussianMetropolis1D, self).__init__(target_distribution, GaussianProposalDistribution(proposal_variance))

    def _get_initial_value(self):
        return random.gauss(0.0, 100)