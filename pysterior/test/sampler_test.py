import matplotlib.pyplot as plt
import numpy as np
from pysterior.energy import MultivariateNormalEnergyClosure
from pysterior.sampler import NUTS
from pysterior import data_model

def visualize_gaussian_direct_sampling():
    # energy = GaussianEnergyClosure(0.0, 5.0)
    energy = MultivariateNormalEnergyClosure(np.array([0.0, 0.0]), np.linalg.inv(np.array([[10,10],[0,10]])))
    samples = NUTS().nuts_with_initial_epsilon(np.array([100.0, 100.0]), energy, 9000, burn_in=100)
    print(samples)
    # print(shapiro(samples))
    plt.plot(*zip(*samples), marker='.', linewidth=0.0)
    plt.show()

class AbstractHamiltonianSampler(object):
    def __init__(self, target_energy=None):
        self.target_energy = target_energy

    #TODO: Acceptance in abstract class

    def sample(self, iterations, burn_in=0, thinning=1):
        raise NotImplementedError


class HamiltonianSamplerStub(AbstractHamiltonianSampler):
    """
    Trivial class used for testing, which does not utilize the target energy.

    Returns samples from a 3D Gaussian distribution with zero mean and unit sphere covariance.
    """
    def sample(self, iterations, burn_in=0, thinning=1):
        samples = data_model.PosteriorSample()
        for i in range(iterations):
            samples.add_sample(np.random.multivariate_normal(np.zeros(3), np.eye(3,3)))
        return samples