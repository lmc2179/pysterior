import unittest
import math

def compose(inner_function, outer_function):
    def composed_function(*args, **kwargs):
        return outer_function(inner_function(*args, **kwargs))
    return composed_function

def subcompose_single(inner_function, target_kwarg, outer_function):
    def composed_function(**kwargs):
        inner_function_result = inner_function(kwargs[target_kwarg])
        kwargs[target_kwarg] = inner_function_result
        return outer_function(**kwargs)
    return composed_function

def sum_map(functions):
    def sum_mapped_function(*args, **kwargs):
        return sum((f(*args, **kwargs)) for f in functions)
    return sum_mapped_function

class FunctionalSpikesTest(unittest.TestCase):
    def test_composition(self):
        square = lambda x: x**2
        double = lambda x: x*2
        square_then_double = compose(square, double)
        double_then_square = compose(double, square)
        self.assertEqual(square_then_double(5), (5**2)*2)
        self.assertEqual(double_then_square(5), (5*2)**2)

    def test_subcomposition(self):
        def add_one(x):
            return x + 1

        def add(x1=None, x2=None):
            return x1 + x2

        add_plus_one = subcompose_single(add_one, 'x1', add)
        self.assertEqual(add(x1=2, x2=2), 4)
        self.assertEqual(add_plus_one(x1=2, x2=2), 5)
        x1_plus_sqrt_x2 = subcompose_single(math.sqrt, 'x2', add)
        self.assertEqual(x1_plus_sqrt_x2(x1=12.3, x2=4), 14.3)


if __name__ == '__main__':
    unittest.main()