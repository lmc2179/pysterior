import unittest
import os
os.path.join('..')
import abstract_differentiable_function
import theano.tensor as T
import theano

class SumOfSquares(abstract_differentiable_function.AbstractDifferentiableFunction):
    def __init__(self):
        X = T.scalar('x1'), T.scalar('x2')
        y = sum([x**2 for x in X])
        self.function = theano.function(X, y)
        self.function_gradient =theano.function(X, theano.grad(y, X))

    def eval(self, x1, x2):
        return self.function(x1, x2)

    def gradient(self, x1, x2):
        return self.function_gradient(x1, x2)

class ConcreteDifferentiableFunctionTest(unittest.TestCase):
    def test_sum_squares(self):
        self.assertEqual(SumOfSquares().eval(1,2),1**2+2**2)
        self.assertEqual(SumOfSquares().gradient(1,2),[2*1, 2*2])