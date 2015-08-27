from theano import tensor as T
import theano
from theano.tensor import nlinalg
from collections import namedtuple
import numpy as np
X = 'X'
MU = 'Mu'

FunctionSpec = namedtuple('FunctionSpec', ['variables', 'output_expression'])

Energy = namedtuple('Energy', ['eval', 'gradient'])

def get_normal_spec():
    X,mu,sigma = [T.vector('X'), T.vector('Mu'), T.matrix('Sigma')]
    GaussianDensitySpec = FunctionSpec(variables=[X, mu, sigma],
                                       output_expression = -0.5*T.dot(T.dot((X-mu).T, nlinalg.matrix_inverse(sigma)), (X-mu)))
    return GaussianDensitySpec

def get_bivariate_normal_spec():
    X1,X2,mu,sigma = [T.scalar('X1'),T.scalar('X2'), T.vector('mu'), T.matrix('sigma')]
    GaussianDensitySpec = FunctionSpec(variables=[X1, X2, mu, sigma],
                                       output_expression = -0.5*T.dot(T.dot((T.concatenate([X1.dimshuffle('x'),X2.dimshuffle('x')])-mu).T,
                                                                            nlinalg.matrix_inverse(sigma)),
                                                                      (T.concatenate([X1.dimshuffle('x'),X2.dimshuffle('x')])-mu)))
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