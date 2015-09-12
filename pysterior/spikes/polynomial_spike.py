import unittest
import numpy as np

def get_inner_move(sequence):
    np_first_zero_index = np.where(sequence==0)[0]
    if np_first_zero_index:
        first_zero = np_first_zero_index[0]
        ones_vector = np.array([1 if i < first_zero else 0
                                for i in range(len(sequence))])
    else:
        ones_vector = np.ones(len(sequence))
    reduced_sequence = sequence - ones_vector
    reshaped_reduced_sequence = get_outer_move(reduced_sequence)
    moved_sequence = reshaped_reduced_sequence + ones_vector
    return moved_sequence




def rindex(seq, target):
    rev = reversed(seq)
    for i, x in enumerate(rev):
        if x == target:
            return len(seq) - 1 - i
    return -1


def get_outer_move(sequence):
    greatest = sequence[0]
    rightmost_greatest_index = rindex(sequence, greatest)
    first_zero = np.where(sequence==0)[0][0]
    sequence[rightmost_greatest_index] -= 1
    sequence[first_zero] += 1
    return sequence

def non_increasing_sequences(l, n):
    initial_sequence = np.array([n] + [0] * (l-1))
    print(initial_sequence)
    seq = initial_sequence
    while seq[0] > 1 and 0 in seq: #TODO: How do we add inner moves to this? What sequence should occur?
        seq = get_outer_move(seq)
        print(seq)

def heap_permutations(L):
    pass

def preprocess_row_polynomial(row, degree, include_bias=True):
    pass

class PolynomialTest(unittest.TestCase):
    def test_seq(self):
        non_increasing_sequences(3, 5)
        print(get_inner_move(np.array([3, 1, 1])))
        print(get_inner_move(np.array([3, 1, 1, 0])))

if __name__ == '__main__':
    unittest.main()