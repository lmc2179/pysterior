from math import log
import random
import abc

import numpy as np
import pyximport;

pyximport.install()
from sampling.norm_pdf import lognormpdf

class AbstractDistribution(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def log_pdf(self, state):
        """The log-likelihood of the target distribution in this state."""

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

class GaussianSphereProposalDistribution(AbstractProposalDistribution):
    def __init__(self, sigma, number_of_parameters):
        self.sigma = sigma
        self.number_of_parameters = number_of_parameters

    def propose(self, current_state):
        return np.array([np.random.normal(x, self.sigma) for x in current_state])

    def transition_log_probability(self, current_state, new_state):
        return 0.0


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
    def __init__(self, proposal_variance, target_distribution, initial_lower=0.0, initial_upper=100.0):
        super(GaussianMetropolis1D, self).__init__(target_distribution, GaussianProposalDistribution(proposal_variance))
        self.initial_lower = initial_lower
        self.initial_upper = initial_upper

    def _get_initial_value(self):
        return random.gauss(self.initial_lower, self.initial_upper)

class RealGaussianDensityParameterSampler(MetropolisHastings): #TODO: Overweight
                                                               #TODO: After breaking up, add subsampling
    def __init__(self, log_pdf, no_of_parameters, data, proposal_variance, hyperprior_precision): #TODO: Too many args
        self.log_pdf = log_pdf
        self.no_of_parameters = no_of_parameters
        target_distribution = self._build_target_distribution(log_pdf, hyperprior_precision, data)
        proposal_distribution = self._build_proposal_distribution(no_of_parameters, proposal_variance)
        super(RealGaussianDensityParameterSampler, self).__init__(target_distribution, proposal_distribution)

    def _build_target_distribution(self, log_pdf, hyperprior_precision, data):
        def log_likelihood(parameter_vector):
            return sum((log_pdf(d, *parameter_vector) for d in data))
        def prior_log_likelihood(parameter_vector):
            return sum((lognormpdf(p, 0, 1.0/hyperprior_precision) for p in parameter_vector))
        def parameter_likelihood(parameter_vector):
            return log_likelihood(parameter_vector) + prior_log_likelihood(parameter_vector)
        return parameter_likelihood

    def _build_proposal_distribution(self, no_of_parameters, proposal_variance):
        return GaussianSphereProposalDistribution(proposal_variance, no_of_parameters)

    def _get_initial_value(self):
        return np.ones(self.no_of_parameters)