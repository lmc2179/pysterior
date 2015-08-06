import theano
import theano.tensor as T

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

class PartiallyDifferentiableEnergyFactory(object):
    pass
    # TODO: This class should
    # Allow the user to construct an energy where a parameters are partitioned into fixed/nondifferentiable and
    # free/differentiable.
    # Essentially, a partial energy function will be returned, and the only input will be the free parameter
    # The idea is that as long as we can evaluate only this parameter, it can be used in a NUTS iteration
    # Without changing the algorithm.
    # As a first test, this will help allow us to treat the X, mu, and sigma of a gaussian in a symmetric way.
    # However, when inferring parameter variables we will also need to include evidence and prior energies.
    # This will be a first step towards a wholemeal treatment of parameter inference.