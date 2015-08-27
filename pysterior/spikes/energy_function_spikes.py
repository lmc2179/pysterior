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

def sum_map(functions):
    def sum_mapped_function(*args, **kwargs):
        return sum((f(*args, **kwargs)) for f in functions)
    return sum_mapped_function

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

class GaussianEvidenceFactory(object):
    def build_evidence_function(self, data):
        gaussian_fxn_spec = energy.get_normal_spec()
        factory = energy.PartiallyDifferentiableFunctionFactory(gaussian_fxn_spec)
        f, mu_gradient = factory.get_partial_diff('mu')
        _, sigma_gradient = factory.get_partial_diff('sigma')
        evidence_fxn = sum_map([functools.partial(f, **{'X': d}) for d in data])
        mu_evidence_gradient = sum_map([functools.partial(mu_gradient, **{'X': d}) for d in data])
        sigma_evidence_gradient = sum_map([functools.partial(sigma_gradient, **{'X': d}) for d in data])
        return evidence_fxn, mu_evidence_gradient, sigma_evidence_gradient

class PriorFactoryTest(unittest.TestCase):
    def test_prior_construction(self):
        factory = SphericalGaussianPriorFactory()
        f, grad = factory.build_prior(1.0, 2)
        self.assertEqual(f(np.array([0,0])), 0.0)
        self.assertEqual(f(np.array([1,1])), -1.0)
        self.assertEqual(list(grad(np.array([0,0]))), [ 0., 0.])
        self.assertEqual(list(grad(np.array([1,1]))), [-1., -1.])

class GaussianEvidenceFactoryTest(unittest.TestCase):
    def test_single_point_evidence(self):
        factory = GaussianEvidenceFactory()
        data = [np.array([0,0])]
        f, mu_grad, sigma_grad = factory.build_evidence_function(data)
        hyp_1 = {'mu': np.array([1.0, 1.0]), 'sigma': np.eye(2)}
        hyp_2 = {'mu': np.array([-1.0, -1.0]), 'sigma': np.eye(2)}
        hyp_0 = {'mu': np.array([0.0, 0.0]), 'sigma': np.eye(2)}
        print(f(**hyp_1), mu_grad(**hyp_1), sigma_grad(**hyp_1))
        print(f(**hyp_2), mu_grad(**hyp_2), sigma_grad(**hyp_2))
        print(f(**hyp_0), mu_grad(**hyp_0), sigma_grad(**hyp_0))


if __name__ == '__main__':
    unittest.main()