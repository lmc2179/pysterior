import unittest
import abstract_differentiable_function
import theano.tensor as T
import os
os.path.join('..')
import bayes_estimator

class Exponential(abstract_differentiable_function.AbstractDifferentiableFunction):
    def _get_variables(self):
        x = T.scalar('x')
        y = x**2
        return [x], [], y

class StochasticGradientDescentTest(unittest.TestCase):
    def univariate_convex_function_test(self):
        optimizer = bayes_estimator.GradientDescent()
        minimum = optimizer.minimize(Exponential(), 10.0, 100, 1e-10)
        print(minimum)
