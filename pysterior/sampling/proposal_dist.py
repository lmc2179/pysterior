import abc
from math import log

import numpy as np


class AbstractProposalDistribution(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        super(AbstractProposalDistribution, self).__init__()

    @abc.abstractmethod
    def transition_log_probability(self, current_state, new_state):
        """Return the log likelihood of a transition under this proposal distribution."""

    @abc.abstractmethod
    def propose(self, current_state):
        """Propose a new state given the current state."""

class MetropolisProposal(AbstractProposalDistribution):
    def transition_log_probability(self, current_state, new_state):
        return log(1.0)

class GaussianMetropolisProposal(MetropolisProposal):
    def __init__(self, sigma, **kwargs):
        self.sigma = sigma
        if np.shape(sigma) == ():
            self._proposal_function = self._propose_1D_normal
        else:
            self._proposal_function = self._propose_multivariate_normal
        super(GaussianMetropolisProposal, self).__init__(**kwargs)

    def _propose_1D_normal(self, current_state):
        return np.random.normal(current_state, self.sigma)

    def _propose_multivariate_normal(self, current_state):
        return np.random.multivariate_normal(current_state, self.sigma)

    def propose(self, current_state):
        return self._proposal_function(current_state)

class GaussianAdaptiveMetropolisProposal(GaussianMetropolisProposal):
    pass #TODO: Rewrite with covariance update