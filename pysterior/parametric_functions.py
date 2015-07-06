import abstract_differentiable_function
import theano.tensor as T

class SupervisedRegressionFunction(abstract_differentiable_function.AbstractDifferentiableFunction):
    """
    Parametric function which is used as the basis for a parametric regression model.
    Inputs and parameters are assumed to be vectors.
    Differentiation is done with respect to the parameters.
    """

    def _get_variables(self):
        x = T.vector('x')
        theta = T.vector('theta')
        y = self._get_output(theta, x)
        return [theta], [x], y

    def _get_output(self, theta, x):
        raise NotImplementedError

    def __call__(self, theta, x):
        return self.eval(theta, x)

class LinearModel(SupervisedRegressionFunction):
    def _get_output(self, theta, x):
        return T.dot(x, theta)