from theano import tensor as T
import theano
from theano.tensor import nlinalg
from collections import namedtuple
import numpy as np
X = 'X'
MU = 'mu'

FunctionSpec = namedtuple('FunctionSpec', ['variables', 'output_expression'])

Energy = namedtuple('Energy', ['eval', 'gradient'])

def get_normal_spec():
    X,mu,sigma = [T.vector('X'), T.vector('mu'), T.matrix('sigma')]
    GaussianDensitySpec = FunctionSpec(variables=[X, mu, sigma],
                                       output_expression = -0.5*T.dot(T.dot((X-mu).T, nlinalg.matrix_inverse(sigma)), (X-mu)))
    return GaussianDensitySpec

class PartiallyDifferentiableFunctionFactory(object):
    def __init__(self, func_spec):
        self.f = theano.function(func_spec.variables,
                                 func_spec.output_expression,
                                 allow_input_downcast=True)
        self.variables = func_spec.variables
        self.var_lookup = {v.name:v for v in func_spec.variables}
        self.output_expression = func_spec.output_expression

    def get_partial_diff(self, differentiable_var_name):
        diff_var = self.var_lookup[differentiable_var_name]
        grad = theano.function(self.variables,
                               theano.grad(self.output_expression,
                                           diff_var),
                               allow_input_downcast=True)
        return self.f, grad

def build_kwarg_closure(f, bound_kwargs):
    def partial_fxn(**kwargs):
        full_kwargs = {}
        full_kwargs.update(bound_kwargs)
        full_kwargs.update(kwargs)
        return f(**full_kwargs)
    return partial_fxn

def build_arg_closure(f, bound_kwargs, unbound_arg_name): #TODO: Add to own class or AbstractEnergyFactory
    def partial_fxn(arg):
        full_kwargs = {}
        full_kwargs.update(bound_kwargs)
        full_kwargs.update({unbound_arg_name: arg})
        return f(**full_kwargs)
    return partial_fxn


class DirectSamplingEnergyFactory(object):
    def construct_energy(self, fxn_spec, bound_arg_dict):
        partial_dif = PartiallyDifferentiableFunctionFactory(fxn_spec)
        f, grad = partial_dif.get_partial_diff(X)
        f_closure = build_arg_closure(f, bound_arg_dict, X)
        grad_closure = build_arg_closure(grad, bound_arg_dict, X)
        return Energy(eval=f_closure,
                      gradient=grad_closure)

class ParameterPosteriorSamplingEnergyFactory(object):
    def construct_energy(self, #TODO: omg so many arguments why
                         fxn_spec,
                         target_param,
                         bound_arg_dict,
                         data,
                         prior_alpha,
                         dimension):
        partial_dif = PartiallyDifferentiableFunctionFactory(fxn_spec)
        f, grad = partial_dif.get_partial_diff(target_param)
        evidence_energies = self._build_evidence_energies(f, grad, bound_arg_dict, target_param, data)
        prior_energy = self._build_prior_energy(f, grad, bound_arg_dict, target_param, prior_alpha)
        total_energy = self._combine_energies(evidence_energies + [prior_energy])