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

class BlockedProposal(MetropolisProposal):
    def __init__(self, proposals=None, block_indices=None):
        self.proposals = proposals
        self.block_indices = block_indices
        self.total_length = sum([len(b) for b in block_indices])
        super(BlockedProposal, self).__init__()

    def _break_into_blocks(self, state):
        return [[state[i] for i in block] for block in self.block_indices]

    def _combine_blocks(self, block_states):
        result = np.zeros(self.total_length)
        for block, indices in zip(block_states, self.block_indices):
            for b,i in zip(block, indices):
                result[i] = b
        return result

    def propose(self, current_state):
        blocks = self._break_into_blocks(current_state)
        proposed_blocks = [block_proposal.propose(block) for block, block_proposal in zip(blocks, self.proposals)]
        return self._combine_blocks(proposed_blocks)