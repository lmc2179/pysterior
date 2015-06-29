import unittest
import abstract_differentiable_function
import theano.tensor as T
import os
os.path.join('..')
import bayes_estimator

class Exponential(abstract_differentiable_function.AbstractDifferentiableFunction):
    def __init__(self, shift):
        self.shift = float(shift)
        super(Exponential, self).__init__()

    def _get_variables(self):
        x = T.scalar('x')
        y = (x+self.shift)**2 + 1
        return [x], [], y

class StochasticGradientDescentTest(unittest.TestCase):
    def univariate_convex_function_test(self):
        optimizer = bayes_estimator.StochasticBatchGradientDescent()
        functions = [Exponential(0), Exponential(3), Exponential(4), Exponential(5)]
        minimum = optimizer.minimize(functions, starting_point=10.2, iterations=1000, batch_size=2)
        self.assertAlmostEqual(minimum, -3.0, delta=0.05)
