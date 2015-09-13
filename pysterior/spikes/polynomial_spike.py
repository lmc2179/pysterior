import unittest
import numpy as np
import itertools
import functools

class NondecreasingFiniteSumFixedSequenceEnumerator(object):
    "Enterprise edition."
    def get_ones_vector(self, sequence):
        if 0 in sequence:
            np_first_zero_index = list(np.where(sequence==0))
        else:
            np_first_zero_index = None
        if np_first_zero_index:
            first_zero = np_first_zero_index[0][0]
            ones_vector = np.array([1 if i < first_zero else 0
                                    for i in range(len(sequence))])
        else:
            min_value = sequence[-1]
            ones_vector = np.ones(len(sequence)) * min_value
        return ones_vector

    def get_inner_move(self, sequence):
        ones_vector = self.get_ones_vector(sequence)
        reduced_sequence = sequence - ones_vector
        reshaped_reduced_sequence = self.get_outer_move(reduced_sequence)
        moved_sequence = reshaped_reduced_sequence + ones_vector
        return moved_sequence

    def rindex(self, seq, target):
        rev = reversed(seq)
        for i, x in enumerate(rev):
            if x == target:
                return len(seq) - 1 - i
        return -1


    def get_outer_move(self, sequence):
        greatest = sequence[0]
        rightmost_greatest_index = self.rindex(sequence, greatest)
        first_zero = np.where(sequence==0)[0][0]
        sequence[rightmost_greatest_index] -= 1
        sequence[first_zero] += 1
        return sequence

    def is_valid(self, seq):
        for l,r in zip(seq[:-1], seq[1:]):
            if r > l:
                return False
        return True

    def inner_move_possible(self, seq):
        ones_vector = self.get_ones_vector(seq)
        reduced_vector = seq - ones_vector
        return self.outer_move_possible(reduced_vector)

    def outer_move_possible(self, sequence):
        has_zero = 0 not in sequence
        if has_zero or sequence[0] == 1 or sequence[0] == 0:
            return False
        return True

    def is_final_config(self, seq):
        return (not self.inner_move_possible(seq)) and (not self.outer_move_possible(seq))

    def non_increasing_sequences(self, l, n):
        initial_sequence = np.array([n] + [0] * (l-1))
        seq = initial_sequence
        sequences = [np.copy(seq)]
        while not self.is_final_config(seq):
            while self.inner_move_possible(seq):
                seq = self.get_inner_move(seq)
                sequences.append(np.copy(seq))
            if self.outer_move_possible(seq):
                seq = self.get_outer_move(seq)
                sequences.append(np.copy(seq))
        return sequences

class PolynomialFeatureGenerator(object):
    def __init__(self, degree, dimension, inclue_bias=True):
        self.degree = degree
        self.include_bias = inclue_bias
        self._set_exponent_vectors(dimension)

    def _set_exponent_vectors(self, size):
        exponent_vectors = []
        if self.include_bias:
            exponent_vectors = [np.zeros(size)]
        for i in range(1,self.degree+1):
            base_exponents = NondecreasingFiniteSumFixedSequenceEnumerator().non_increasing_sequences(size, i)
            all_exponents_nested = [list(set(itertools.permutations(exponent))) for exponent in base_exponents]
            all_exponents = list(itertools.chain.from_iterable(all_exponents_nested))
            vectorized_exponents = [np.array(x) for x in all_exponents]
            exponent_vectors.extend(vectorized_exponents)
        self.exponent_vectors = exponent_vectors

    def _get_polynomial_term(self, x, exponents):
        product_terms = [base**ex for base,ex in zip(x, exponents)]
        product = lambda x1, x2: x1*x2
        return functools.reduce(product, product_terms)

    def preprocess(self, row):
        make_term_from_exponents = functools.partial(self._get_polynomial_term, row)
        poly_row = np.array(list(map(make_term_from_exponents, self.exponent_vectors)))
        return poly_row

class SequenceTest(unittest.TestCase):
    def test_seq(self):
        sequences = NondecreasingFiniteSumFixedSequenceEnumerator().non_increasing_sequences(5, 3)
        expected = ((3, 0, 0, 0, 0), (2, 1, 0, 0, 0), (1, 1, 1, 0, 0))
        result_set = set(tuple([tuple(s) for s in sequences]))
        self.assertEqual(set(expected), result_set)

    def test_seq_2(self):
        sequences = NondecreasingFiniteSumFixedSequenceEnumerator().non_increasing_sequences(3, 3)
        expected = ((3, 0, 0), (2, 1, 0), (1, 1, 1))
        result_set = set(tuple([tuple(s) for s in sequences]))
        self.assertEqual(set(expected), result_set)

    def test_seq_3(self):
        sequences = NondecreasingFiniteSumFixedSequenceEnumerator().non_increasing_sequences(2, 3)
        expected = ((3, 0), (2, 1))
        result_set = set(tuple([tuple(s) for s in sequences]))
        self.assertEqual(set(expected), result_set)

    def test_seq_4(self):
        sequences = NondecreasingFiniteSumFixedSequenceEnumerator().non_increasing_sequences(1, 3)
        expected = {(3,)}
        result_set = set(tuple([tuple(s) for s in sequences]))
        self.assertEqual(expected, result_set)

    def test_quadratic(self):
        x = np.array([2,5])
        gen = PolynomialFeatureGenerator(2, len(x))
        self.assertEqual(list(gen.preprocess(x)), [1, 5, 2, 4, 25, 10])

if __name__ == '__main__':
    unittest.main()