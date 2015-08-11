from theano import tensor as T
import theano
from collections import namedtuple

FunctionSpec = namedtuple('FunctionSpec', ['variables', 'output_expression'])

Energy = namedtuple('Energy', ['eval', 'grad'])

def get_unit_normal_spec():
    X,mu = [T.vector('X'), T.vector('mu')]
    UnitSphereGaussianDensitySpec = FunctionSpec(variables=[X, mu],
                                       output_expression =  -0.5*T.dot((X-mu).T, (X-mu)))
    return UnitSphereGaussianDensitySpec

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

class AbstractEnergyClosure(object):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.delegate_energy = self._get_delegate_energy()

    def _get_delegate_energy(self):
        raise NotImplementedError

    def eval(self, x):
        return self.delegate_energy.eval(x, self.mu, self.sigma)

    def gradient(self, x):
        return self.delegate_energy.gradient(x, self.mu, self.sigma)

class AbstractDifferentiableFunction(object):
    def __init__(self):
        differentiable_argument, other_arguments, output = self._get_variables()
        all_arguments = differentiable_argument + other_arguments
        self.function = theano.function(all_arguments, output, allow_input_downcast=True)
        self.function_gradient =theano.function(all_arguments, theano.grad(output, differentiable_argument),
                                                allow_input_downcast=True)

    def _get_variables(self):
        """Returns a tuple of:
            differential_argument: a vector or list of scalars
            other_arguments: list of theano variables
            output: a function of the arguments
            """
        raise NotImplementedError

    def eval(self, *args):
        "Evaluate the function - arguments are assumed to be the same order as in _get_variables."
        return self.function(*args)

    def gradient(self, *args):
        "Evaluate the function's gradient - arguments are assumed to be the same order as in _get_variables."
        return self.function_gradient(*args)[0]

class GaussianEnergy(AbstractDifferentiableFunction): #TODO: Fix these
    def _get_variables(self):
        X = T.scalar('x')
        mu = T.scalar('mu')
        sigma = T.scalar('sigma')
        y = -(X - mu)**2 * (1.0/sigma)
        return [X], [mu, sigma], y


class MultivariateNormalEnergy(AbstractDifferentiableFunction):#TODO: Fix these
    def _get_variables(self):
        X = T.vector('x')
        mu = T.vector('mu')
        sigma = T.matrix('sigma')
        inv_covariance = sigma
        y = -0.5*T.dot(T.dot((X-mu).T, inv_covariance), (X-mu))
        return [X], [mu, sigma], y


class MultivariateNormalEnergyClosure(AbstractEnergyClosure): #TODO: Deprecate these
    def _get_delegate_energy(self):
        return MultivariateNormalEnergy()


class GaussianEnergyClosure(AbstractEnergyClosure): #TODO: Deprecate these
    def _get_delegate_energy(self):
        return GaussianEnergy()