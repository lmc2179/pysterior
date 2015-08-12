import numpy as np
import matplotlib.pyplot as plt

from energy import DirectSamplingFactory
from pysterior import energy
from pysterior import sampler

def visualize_spherical_gaussian_direct_sampling():
    E = DirectSamplingFactory().construct_energy(energy.get_normal_spec(np.eye(2)),
                                         {'mu': np.array([0.0, 0.0])})
    samples = sampler.NUTS().nuts_with_initial_epsilon(np.array([100.0, 100.0]), E, 9000, burn_in=100)
    print(samples)
    plt.plot(*zip(*samples), marker='.', linewidth=0.0)
    plt.show()

def visualize_gaussian_direct_sampling():
    E = DirectSamplingFactory().construct_energy(energy.get_normal_spec(np.array([[10,10],[0,10]])),
                                         {'mu': np.array([0.0, 0.0])})
    samples = sampler.NUTS().nuts_with_initial_epsilon(np.array([100.0, 100.0]), E, 9000, burn_in=100)
    print(samples)
    plt.plot(*zip(*samples), marker='.', linewidth=0.0)
    plt.show()

visualize_gaussian_direct_sampling()