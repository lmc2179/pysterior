from pysterior import gibbs_sampler, energy
import matplotlib.pyplot as plt
import numpy as np

def visualize_gaussian_direct_sampling():
    gaussian_fxn_spec = energy.get_bivariate_normal_spec()
    sampler = gibbs_sampler.GibbsDirectSampler(gaussian_fxn_spec, ['X1', 'X2'], {'mu': np.array([0.0, 0.0]),
                                                                                'sigma': np.array([[10,10],[0,10]])})
    samples = sampler.run_sampling(iterations=1000,
                                                                                             initial_point=[100.0, -100.0])

    print(samples)
    plt.plot(*zip(*samples), marker='.', linewidth=0.0)
    plt.show() #TODO: This works for the most part, but the epsilon heuristic keeps exploding

visualize_gaussian_direct_sampling()
