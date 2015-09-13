import unittest
import numpy as np
from pysterior.regression import _NondecreasingSequenceEnumerator, _PolynomialFeatureGenerator


class SequenceTest(unittest.TestCase):
    def test_seq(self):
        sequences = _NondecreasingSequenceEnumerator().non_increasing_sequences(5, 3)
        expected = ((3, 0, 0, 0, 0), (2, 1, 0, 0, 0), (1, 1, 1, 0, 0))
        result_set = set(tuple([tuple(s) for s in sequences]))
        self.assertEqual(set(expected), result_set)

    def test_seq_2(self):
        sequences = _NondecreasingSequenceEnumerator().non_increasing_sequences(3, 3)
        expected = ((3, 0, 0), (2, 1, 0), (1, 1, 1))
        result_set = set(tuple([tuple(s) for s in sequences]))
        self.assertEqual(set(expected), result_set)

    def test_seq_3(self):
        sequences = _NondecreasingSequenceEnumerator().non_increasing_sequences(2, 3)
        expected = ((3, 0), (2, 1))
        result_set = set(tuple([tuple(s) for s in sequences]))
        self.assertEqual(set(expected), result_set)

    def test_seq_4(self):
        sequences = _NondecreasingSequenceEnumerator().non_increasing_sequences(1, 3)
        expected = {(3,)}
        result_set = set(tuple([tuple(s) for s in sequences]))
        self.assertEqual(expected, result_set)

    def test_quadratic(self):
        x = np.array([2,5])
        gen = _PolynomialFeatureGenerator(2, len(x))
        self.assertEqual(list(gen.preprocess(x)), [1, 5, 2, 4, 25, 10])

    def test_quadratic_univariate(self):
        x = np.array(5)
        gen = _PolynomialFeatureGenerator(2, 1)
        self.assertEqual(list(gen.preprocess(x)), [1, 5, 25])

if __name__ == '__main__':
    unittest.main()