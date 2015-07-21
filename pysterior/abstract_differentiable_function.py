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
