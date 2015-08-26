from pysterior import energy
import functools
import numpy as np

class EnergyBuilder(object):
    def build_energy(self, *args, **kwargs):
        raise NotImplementedError

class GaussianDirectEnergyBuilder(EnergyBuilder):
    def build_energy(self, TRUE_MU, TRUE_SIGMA):
        gaussian_fxn_spec = energy.get_bivariate_normal_spec()
        target_variables = [v.name for v in gaussian_fxn_spec.variables[0:2]]
        factory = energy.PartiallyDifferentiableFunctionFactory(gaussian_fxn_spec)
        f, f_x1 = factory.get_partial_diff('X1')
        _, f_x2 = factory.get_partial_diff('X2')
        closed_variables = {'mu': TRUE_MU, 'sigma': TRUE_SIGMA}
        closed_total_energy = functools.partial(f, **closed_variables)
        closed_gradients = [functools.partial(f_x1, **closed_variables), functools.partial(f_x2, **closed_variables)]
        return closed_total_energy, target_variables, closed_gradients