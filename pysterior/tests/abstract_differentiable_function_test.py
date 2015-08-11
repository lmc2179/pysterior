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

def _get_fxn(x1=None, x2=None):
    return T.dot(x1, x2)

class DotProduct(abstract_differentiable_function.AbstractDifferentiableFunction):
    def _get_variables(self):
        x1 = T.vector('x1')
        x2 = T.vector('x2')
        y = _get_fxn(x1=x1, x2=x2)
        return [x1], [x2], y

class DotProductClosure(abstract_differentiable_function.AbstractDifferentiableFunction):
    def __init__(self, x2):
        super(DotProductClosure, self).__init__()
        self.x2=x2

    def _get_variables(self):
        x1 = T.vector('x1')
        y = _get_fxn(x1=x1, x2=self.x2) #TODO: This doesn't work, because at compile time x2 is not defined
        return [x1], [], y

class ConcreteDifferentiableFunctionTest(unittest.TestCase):
    def test_sum_squares(self):
        self.assertEqual(SumOfSquares().eval(1,2),1**2+2**2)
        self.assertEqual(SumOfSquares().gradient(1,2),[2*1, 2*2])

    def test_dot_product(self):
        self.assertEqual(DotProduct().eval([1,1], [1,1]), 2.0)
        assert T.eq(DotProduct().gradient([1,1], [1,1]) , [1.0, 1.0])

    def test_dot_product_closure(self):
        dpc = DotProductClosure([1,1])
        self.assertEqual(dpc.eval([1,1]), 2.0)
        assert T.eq(dpc.gradient([1,1]) , [1.0, 1.0])