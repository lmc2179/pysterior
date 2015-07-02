import unittest
import random
from pysterior.data_model import PosteriorSample

class PosteriorSampleTest(unittest.TestCase):
    TEST_DATA_SIZE = 100
    def test_containment(self):
        data = [random.randint(0,100) for i in range(self.TEST_DATA_SIZE)]
        sample = PosteriorSample()
        [sample.add_sample(d) for d in data]
        self.assertEqual(set(sample.get_samples()), set(data))

    def test_average(self):
        data = [random.randint(0,100) for i in range(self.TEST_DATA_SIZE)]
        sample = PosteriorSample()
        [sample.add_sample(d) for d in data]
        data_avg = (1.0*sum(data))/len(data)
        self.assertAlmostEqual(data_avg, sample.get_mean(), delta=1e-12)

    def test_median_odd(self):
        data = [0,1,2,3,4,5,6]
        sample = PosteriorSample()
        [sample.add_sample(d) for d in data]
        data_median = 3
        self.assertEqual(data_median, sample.get_median())

    def test_median_even(self):
        data = [0,1,2,3,4,5,6,7]
        sample = PosteriorSample()
        [sample.add_sample(d) for d in data]
        data_median = 3.5
        self.assertEqual(data_median, sample.get_median())
