import unittest
from pysterior import gibbs_sampler, energy
import functools
import numpy as np
import matplotlib.pyplot as plt

class MvNormalGibbsSamplingTest(unittest.TestCase):
    def test_sampling(self):
        gaussian_fxn_spec = energy.get_bivariate_normal_spec()
        target_variables = [v.name for v in gaussian_fxn_spec.variables[0:2]]
        factory = energy.PartiallyDifferentiableFunctionFactory(gaussian_fxn_spec)
        f, f_x1 = factory.get_partial_diff('X1')
        _, f_x2 = factory.get_partial_diff('X2')
        TRUE_MU = np.array([1,2])
        TRUE_SIGMA = np.array([[10,10],[0,10]])
        closed_variables = {'mu': TRUE_MU, 'sigma': TRUE_SIGMA}
        closed_total_energy = functools.partial(f, **closed_variables)
        closed_gradients = [functools.partial(f_x1, **closed_variables), functools.partial(f_x2, **closed_variables)]
        sampler = gibbs_sampler.GibbsSampler(target_variables,
                                             closed_total_energy,
                                             closed_gradients)
        initial_state = {'X1': 1.0, 'X2': 2.0}
        samples = sampler.run_sampling(initial_state, iterations=1000, burn_in=20)
        plt.plot(*zip(*samples), linewidth=0.0, marker='.')
        plt.show()

if __name__ == '__main__':
    unittest.main()