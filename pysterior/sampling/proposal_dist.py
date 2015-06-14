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
    def __init__(self, sigma=None):
        self.sigma = sigma

    def propose(self, current_state):
        return np.random.normal(current_state, self.sigma)

class SphereGaussianMetropolisProposal(MetropolisProposal):
    def __init__(self, sigma=None, number_of_parameters=None):
        self.sigma = sigma
        self.number_of_parameters = number_of_parameters

    def propose(self, current_state):
        return np.array([np.random.normal(x, self.sigma) for x in current_state])


class StatefulProposalMixin(AbstractProposalDistribution):
    """
    Mixins are used to introduce state in a reusable way.

    All Mixins should inherit from StatefulProposalMixin and should not define any methods other than update_state(s)
    and set_initial_state().

    They should clearly indicate what state is introduced.
    """
    __metaclass__ = abc.ABCMeta
    def __init__(self, **kwargs):
        self._initialize_state(**kwargs)
        super(StatefulProposalMixin, self).__init__(**kwargs)

    def propose(self, current_state):
        self._update(current_state)
        return super(StatefulProposalMixin, self).propose(current_state)

    @abc.abstractmethod
    def _update(self, current_state):
        "Update the mixin's state. This is called before each new state is proposed."

    @abc.abstractmethod
    def _initialize_state(self, **init_kwargs):
        "Initialize the mixin's state. This is called first in the __init__."

class RejectionRateMixin(StatefulProposalMixin):
    def _initialize_state(self, **init_kwargs):
        self._previous_state = None
        self._rejected_sample_count = 0
        self._total_sample_count = 0

    def _update(self, current_state):
        if self._previous_state is None: #This is the first state we have observed
            self._rejected_sample_count = 0
        else:
            if self._previous_state == current_state:
                self._rejected_sample_count += 1
            self._total_sample_count += 1
        self._previous_state = current_state

    def get_rejection_rate(self):
        return (1.0*self._rejected_sample_count) / self._total_sample_count