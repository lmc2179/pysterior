import theano.tensor as T
import theano
from collections import namedtuple

FunctionSpec = namedtuple('FunctionSpec', ['variables', 'output_expression'])

class PartiallyDifferentiableFunctionFactory(object):
    def __init__(self, func_spec):
        self.f = theano.function(func_spec.variables,
                                 func_spec.output_expression)
        self.variables = func_spec.variables
        self.var_lookup = {v.name:v for v in func_spec.variables}
        self.output_expression = func_spec.output_expression

    def get_partial_diff(self, differentiable_var_name):
        diff_var = self.var_lookup[differentiable_var_name]
        grad = theano.function(self.variables,
                               theano.grad(self.output_expression,
                                           diff_var))
        return self.f, grad

def build_kwarg_closure(f, bound_kwargs):
    def partial_fxn(**kwargs):
        full_kwargs = {}
        full_kwargs.update(bound_kwargs)
        full_kwargs.update(kwargs)
        return f(**full_kwargs)
    return partial_fxn

#Note kwargs implicit, see f.input_storage
#TODO: Write an energy generator, which will produce both direct sampling and parameter evidence energies for gaussians
#TODO: Unit tests
#TODO: Supervised learning