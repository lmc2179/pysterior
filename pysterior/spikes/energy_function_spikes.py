from pysterior import energy
import functools
import numpy as np
import unittest

def build_arg_closure(f, bound_kwargs, unbound_arg_name):
    def partial_fxn(arg):
        full_kwargs = {}
        full_kwargs.update(bound_kwargs)
        full_kwargs.update({unbound_arg_name: arg})
        return f(**full_kwargs)
    return partial_fxn

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

class SphericalGaussianPriorFactory(object):
    def build_prior(self, alpha, dimension):
        gaussian_fxn_spec = energy.get_normal_spec()
        factory = energy.PartiallyDifferentiableFunctionFactory(gaussian_fxn_spec)
        f, gradient = factory.get_partial_diff('X')
        closed_variables = {'mu': np.zeros(dimension),
                            'sigma': np.eye(dimension) * alpha}
        closed_total_energy = build_arg_closure(f, closed_variables, 'X')
        closed_gradient = build_arg_closure(gradient, closed_variables, 'X')
        return closed_total_energy, closed_gradient

class PriorFactoryTest(unittest.TestCase):
    def test_prior_construction(self):
        factory = SphericalGaussianPriorFactory()
        f, grad = factory.build_prior(1.0, 2)
        self.assertEqual(f(np.array([0,0])), 0.0)
        self.assertEqual(f(np.array([1,1])), -1.0)
        self.assertEqual(list(grad(np.array([0,0]))), [ 0., 0.])
        self.assertEqual(list(grad(np.array([1,1]))), [-1., -1.])

if __name__ == '__main__':
    unittest.main()