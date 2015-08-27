from pysterior import energy
import functools
import numpy as np
import unittest

def subcompose_single(inner_function, target_kwarg, outer_function):
    def composed_function(**kwargs):
        inner_function_result = inner_function(kwargs[target_kwarg])
        kwargs[target_kwarg] = inner_function_result
        return outer_function(**kwargs)
    return composed_function

def flatten(x):
    if len(np.shape(x)) > 0:
        return x.flatten()
    return x

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
        flatten_f = subcompose_single(flatten, 'X', f)
        flatten_gradient = subcompose_single(flatten, 'X', gradient)
        closed_total_energy = build_arg_closure(flatten_f, closed_variables, 'X')
        closed_gradient = build_arg_closure(flatten_gradient, closed_variables, 'X')
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

class GaussianTotalEnergy(object):
    #TODO: Refactor this to eliminate duplication, it is truly abominable
    def __init__(self, mu_alpha, sigma_alpha, data):
        dimension = len(data[0])
        self.mu_prior, self.mu_prior_gradient = SphericalGaussianPriorFactory().build_prior(mu_alpha, dimension)
        self.sigma_prior, self.sigma_prior_gradient = SphericalGaussianPriorFactory().build_prior(sigma_alpha, dimension**2)
        self.evidence_function, self.mu_evidence_gradient, self.sigma_evidence_gradient = GaussianEvidenceFactory().build_evidence_function(data)

    def eval_total_energy(self, state):
        evidence_energy = self.evidence_function(**state)
        mu_prior_energy = self.mu_prior(state['mu'])
        sigma_prior_energy = self.sigma_prior(state['sigma'])
        return evidence_energy + mu_prior_energy + sigma_prior_energy

    def eval_energy_mu_gradient(self, state):
        evidence_energy = self.mu_evidence_gradient(**state)
        mu_prior_energy = self.mu_prior_gradient(state['mu'])
        sigma_prior_energy = 0.0
        return evidence_energy + mu_prior_energy + sigma_prior_energy

    def eval_energy_sigma_gradient(self, state):
        evidence_energy = self.sigma_evidence_gradient(**state)
        mu_prior_energy = 0.0
        sigma_prior_energy = self.sigma_prior(state['sigma'])
        return evidence_energy + mu_prior_energy + sigma_prior_energy


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
        print('Single point: ')
        factory = GaussianEvidenceFactory()
        data = [np.array([0,0])]
        f, mu_grad, sigma_grad = factory.build_evidence_function(data)
        hyp_1 = {'mu': np.array([1.0, 1.0]), 'sigma': np.eye(2)}
        hyp_2 = {'mu': np.array([-1.0, -1.0]), 'sigma': np.eye(2)}
        hyp_0 = {'mu': np.array([0.0, 0.0]), 'sigma': np.eye(2)}
        print(f(**hyp_1), mu_grad(**hyp_1), sigma_grad(**hyp_1))
        print(f(**hyp_2), mu_grad(**hyp_2), sigma_grad(**hyp_2))
        print(f(**hyp_0), mu_grad(**hyp_0), sigma_grad(**hyp_0))

    def test_multi_point_evidence(self):
        print('Multi point: ')
        factory = GaussianEvidenceFactory()
        data = [np.array([1,0]),np.array([-1,0]),np.array([0,1]),np.array([0,-1])]
        f, mu_grad, sigma_grad = factory.build_evidence_function(data)
        hyp_1 = {'mu': np.array([1.0, 1.0]), 'sigma': np.eye(2)}
        hyp_2 = {'mu': np.array([-1.0, -1.0]), 'sigma': np.eye(2)}
        hyp_0 = {'mu': np.array([0.0, 0.0]), 'sigma': np.eye(2)}
        print(f(**hyp_1), mu_grad(**hyp_1), sigma_grad(**hyp_1))
        print(f(**hyp_2), mu_grad(**hyp_2), sigma_grad(**hyp_2))
        print(f(**hyp_0), mu_grad(**hyp_0), sigma_grad(**hyp_0))


class GaussianTotalEnergyTest(unittest.TestCase):
    def test_end_to_end(self):
        print('GAUSSIAN TOTAL ENERGY: ')
        E = GaussianTotalEnergy(1.0, 1.0, [np.array([0,0])])
        state = {'mu': np.array([0.0, 1.0]), 'sigma': np.eye(2)}
        print(E.eval_energy_mu_gradient(state))
        state = {'mu': np.array([1.0, 0.0]), 'sigma': np.eye(2)}
        print(E.eval_energy_mu_gradient(state))
        state = {'mu': np.array([0.0, 0.0]), 'sigma': np.eye(2)}
        print(E.eval_energy_mu_gradient(state))

if __name__ == '__main__':
    unittest.main()