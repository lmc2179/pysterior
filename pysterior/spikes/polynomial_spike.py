import unittest
import numpy as np

class NondecreasingFiniteSumFixedSequenceEnumerator(object):
    "This one's for you, Java programmers."
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
        while not self.is_final_config(seq): #TODO: How do we add inner moves to this? What sequence should occur?
            while self.inner_move_possible(seq):
                seq = self.get_inner_move(seq)
                sequences.append(np.copy(seq))
            if self.outer_move_possible(seq):
                seq = self.get_outer_move(seq)
                sequences.append(np.copy(seq))
        return sequences



def heap_permutations(L):
    pass

def preprocess_row_polynomial(row, degree, include_bias=True):
    pass

class PolynomialTest(unittest.TestCase):
    def test_seq(self):
        print(NondecreasingFiniteSumFixedSequenceEnumerator().non_increasing_sequences(4, 3))
        # print(get_inner_move(np.array([3, 1, 1])))
        # print(get_inner_move(np.array([3, 1, 1, 0])))

if __name__ == '__main__':
    unittest.main()