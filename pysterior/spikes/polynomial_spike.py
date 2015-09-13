import unittest
import numpy as np

def get_ones_vector(sequence):
    np_first_zero_index = list(np.where(sequence==0))
    if np_first_zero_index:
        first_zero = np_first_zero_index[0][0]
        ones_vector = np.array([1 if i < first_zero else 0
                                for i in range(len(sequence))])
    else:
        ones_vector = np.ones(len(sequence))
    return ones_vector

def get_inner_move(sequence):
    ones_vector = get_ones_vector(sequence)
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

def is_valid(seq):
    for l,r in zip(seq[:-1], seq[1:]):
        if r > l:
            return False
    return True

def inner_move_possible(seq):
    ones_vector = get_ones_vector(seq)
    reduced_vector = seq - ones_vector
    return outer_move_possible(reduced_vector)

def outer_move_possible(sequence):
    has_zero = 0 not in sequence
    if has_zero or sequence[0] == 1:
        return False
    return True

def is_final_config(seq):
    return (not inner_move_possible(seq)) and (not outer_move_possible(seq))

def non_increasing_sequences(l, n):
    initial_sequence = np.array([n] + [0] * (l-1))
    print(initial_sequence)
    seq = initial_sequence
    while not is_final_config(seq): #TODO: How do we add inner moves to this? What sequence should occur?
        while inner_move_possible(seq):
            seq = get_inner_move(seq)
            print(seq)
        if outer_move_possible(seq):
            seq = get_outer_move(seq)
            print(seq)



def heap_permutations(L):
    pass

def preprocess_row_polynomial(row, degree, include_bias=True):
    pass

class PolynomialTest(unittest.TestCase):
    def test_seq(self):
        print(non_increasing_sequences(3, 5))
        # print(get_inner_move(np.array([3, 1, 1])))
        # print(get_inner_move(np.array([3, 1, 1, 0])))

if __name__ == '__main__':
    unittest.main()