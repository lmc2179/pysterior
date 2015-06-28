import unittest
import os
os.path.join('..')
import abstract_differentiable_function
import theano.tensor as T

class SumOfSquares(abstract_differentiable_function.AbstractDifferentiableFunction):
    def _get_variables(self):
        X = [T.scalar('x1'), T.scalar('x2')]
        y = sum([x**2 for x in X])
        return X, [], y

class ConcreteDifferentiableFunctionTest(unittest.TestCase):
    def test_sum_squares(self):
        self.assertEqual(SumOfSquares().eval(1,2),1**2+2**2)
        self.assertEqual(SumOfSquares().gradient(1,2),[2*1, 2*2])
