import os
os.path.join('..')
import unittest
from archive import energy
import theano.tensor as T

class PartiallyDiffFunctionFactoryTest(unittest.TestCase):
    def test_product(self):
        a,b = T.scalar('a'), T.scalar('b')
        product = a*b
        product_spec = energy.FunctionSpec(variables=[a,b],
                                                               output_expression=product)
        factory = energy.PartiallyDifferentiableFunctionFactory(product_spec)
        f, a_grad = factory.get_partial_diff('a')
        self.assertEqual(f(a=2, b=2), 4)
        self.assertEqual(a_grad(a=1.0, b=2.0), 2.0)

class TestClosure(unittest.TestCase):
    def test_sum_kwarg_closure(self):
        def sum_dict(a, b):
            return a+b

        add_one = energy.build_kwarg_closure(sum_dict, {'a': 1})
        self.assertEqual(add_one(b=100), 101)

    def test_sum_arg_closure(self):
        def sum_dict(a, b):
            return a+b

        add_one = energy.build_arg_closure(sum_dict, {'a': 1}, 'b')
        self.assertEqual(add_one(100), 101)

if __name__ == '__main__':
    unittest.main()