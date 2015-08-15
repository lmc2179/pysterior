import numpy as np
import matplotlib.pyplot as plt

from energy import DirectSamplingEnergyFactory, ParameterPosteriorSamplingEnergyFactory
from pysterior import energy
from pysterior import sampler
import random

def visualize_gaussian_direct_sampling():
    E = DirectSamplingEnergyFactory().construct_energy(energy.get_normal_spec(),
                                         {'mu': np.array([0.0, 0.0]),
                                          'sigma': np.array([[10,10],[0,10]])})
    samples = sampler.NUTS().nuts_with_initial_epsilon(np.array([100.0, 100.0]), E, 9000, burn_in=100)
    print(samples)
    plt.plot(*zip(*samples), marker='.', linewidth=0.0)
    plt.show()

def visualize_gaussian_posterior_sampling():
    f = ParameterPosteriorSamplingEnergyFactory()
    E = f.construct_energy(energy.get_normal_spec(),
                          'mu',
                        {'sigma':np.array([[10,10],[0,10]])},
                        [np.array([random.gauss(0,1),random.gauss(0,1)]) for i in range(100)])
    samples = sampler.NUTS().nuts_with_initial_epsilon(np.array([100.0, 100.0]), E, 9000, burn_in=100)
    print(samples)
    plt.plot(*zip(*samples), marker='.', linewidth=0.0)
    plt.show()

visualize_gaussian_direct_sampling()