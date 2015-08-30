import unittest
import functools

import numpy as np
import matplotlib.pyplot as plt

from archive import gibbs_sampler, energy
from pysterior.spikes import energy_function_spikes


class MvNormalGibbsSamplingTest(unittest.TestCase):
    def _run_gaussian_direct_sampling(self, TRUE_MU, TRUE_SIGMA):
        gaussian_fxn_spec = energy.get_bivariate_normal_spec()
        target_variables = [v.name for v in gaussian_fxn_spec.variables[0:2]]
        factory = energy.PartiallyDifferentiableFunctionFactory(gaussian_fxn_spec)
        f, f_x1 = factory.get_partial_diff('X1')
        _, f_x2 = factory.get_partial_diff('X2')
        closed_variables = {'mu': TRUE_MU, 'sigma': TRUE_SIGMA}
        closed_total_energy = functools.partial(f, **closed_variables)
        closed_gradients = [functools.partial(f_x1, **closed_variables), functools.partial(f_x2, **closed_variables)]
        sampler = gibbs_sampler.GibbsSampler(target_variables,
                                             closed_total_energy,
                                             closed_gradients)
        initial_state = {'X1': 1.0, 'X2': 2.0}
        samples = sampler.run_sampling(initial_state, iterations=1000, burn_in=20)
        return samples

    @unittest.skip('')
    def test_sampling(self):
        TRUE_MU = np.array([1,2])
        TRUE_SIGMA = np.array([[10,10],[0,10]])
        samples = self._run_gaussian_direct_sampling(TRUE_MU, TRUE_SIGMA)
        plt.plot(*zip(*samples), linewidth=0.0, marker='.')
        plt.show()

    def test_gaussian_posterior(self):
        gaussian_fxn_spec = energy.get_normal_spec()
        target_variables = [v.name for v in gaussian_fxn_spec.variables[1:]]
        data = [np.array([0,0]), np.array([1,0]), np.array([-1,0]), np.array([0,1]), np.array([0,-1])]
        E = energy_function_spikes.GaussianTotalEnergy(10000.0, 10000.0, data)
        sampler = gibbs_sampler.GibbsSampler(target_variables,
                                             E.eval_total_energy,
                                             [E.eval_energy_mu_gradient, E.eval_energy_sigma_gradient])
        initial_state = {'Mu': np.array([100,100]), 'Sigma': np.array([20, 10, 10, 15])}
        samples = sampler.run_sampling(initial_state, iterations=10, burn_in=20)
    

if __name__ == '__main__':
    unittest.main()