from theano import tensor as T
import theano
from collections import namedtuple
import numpy as np
X = 'X'
MU = 'mu'

FunctionSpec = namedtuple('FunctionSpec', ['variables', 'output_expression'])

Energy = namedtuple('Energy', ['eval', 'gradient'])

def get_normal_spec(covariance_matrix):
    X,mu = [T.vector('X'), T.vector('mu')]
    inv_covariance = np.linalg.inv(covariance_matrix)
    GaussianDensitySpec = FunctionSpec(variables=[X, mu],
                                       output_expression =  -0.5*T.dot(T.dot((X-mu).T, inv_covariance), (X-mu)))
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

def build_arg_closure(f, bound_kwargs, unbound_arg_name):
    def partial_fxn(arg):
        full_kwargs = {}
        full_kwargs.update(bound_kwargs)
        full_kwargs.update({unbound_arg_name: arg})
        return f(**full_kwargs)
    return partial_fxn


class DirectSamplingFactory(object):
    def construct_energy(self, fxn_spec, bound_arg_dict):
        partial_dif = PartiallyDifferentiableFunctionFactory(fxn_spec)
        f, grad = partial_dif.get_partial_diff(X)
        f_closure = build_arg_closure(f, bound_arg_dict, X)
        grad_closure = build_arg_closure(grad, bound_arg_dict, X)
        return Energy(eval=f_closure,
                      gradient=grad_closure)