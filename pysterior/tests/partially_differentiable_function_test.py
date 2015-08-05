import os
os.path.join('..')
import unittest
import partially_differentiable_function
import theano.tensor as T

class PartiallyDiffFunctionFactoryTest(unittest.TestCase):
    def test_product(self):
        a,b = T.scalar('a'), T.scalar('b')
        product = a*b
        product_spec = partially_differentiable_function.FunctionSpec(variables=[a,b],
                                                               output_expression=product)
        factory = partially_differentiable_function.PartiallyDifferentiableFunctionFactory(product_spec)
        f, a_grad = factory.get_partial_diff('a')
        self.assertEqual(f(a=2, b=2), 4)
        self.assertEqual(a_grad(a=1.0, b=2.0), 2.0)

class TestClosure(unittest.TestCase):
    def test_sum(self):
        def sum_dict(a, b):
            return a+b

        add_one = partially_differentiable_function.build_kwarg_closure(sum_dict, {'a': 1})
        self.assertEqual(add_one(b=100), 101)


class EnergyClosureFactorytest(unittest.TestCase):
    pass #Direct sampling of gaussian vs prior+evidence sampling

if __name__ == '__main__':
    unittest.main()