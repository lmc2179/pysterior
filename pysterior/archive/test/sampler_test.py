import unittest

import theano.tensor as T
from theano import function, grad
import matplotlib.pyplot as plt
from scipy.stats import shapiro

from archive import energy

from archive.sampler import NUTS
from pysterior import energy



#TODO: Make these automated tests

class TestUnivariateDistributions(unittest.TestCase):
    @unittest.skip('')
    def test_normal(self):
        TRUE_MEAN = -10.0
        TRUE_SIGMA = 3.7
        sampler = NUTS()
        X = T.scalar('X')
        output = -0.5 * ((X - TRUE_MEAN)**2) * (TRUE_SIGMA ** (-2))
        f = function([X], output, allow_input_downcast=True)
        grad_f = function([X],
                          grad(output, X),
                          allow_input_downcast=True)
        E = energy.Energy(eval=f,
                          gradient=grad_f)
        samples = sampler.nuts_with_initial_epsilon(0.0, E, iterations=10000, burn_in=100)
        w,p = shapiro(samples)
        self.assertGreater(p, 0.01)

    def test_laplace(self):
        TRUE_MEAN = -10.0
        TRUE_SCALE = 0.07
        sampler = NUTS()
        X = T.scalar('X')
        output = -1.0 * (abs(X - TRUE_MEAN) / TRUE_SCALE)
        f = function([X], output, allow_input_downcast=True)
        grad_f = function([X],
                          grad(output, X),
                          allow_input_downcast=True)
        E = energy.Energy(eval=f,
                          gradient=grad_f)
        samples = sampler.nuts_with_initial_epsilon(0.0, E, iterations=10000, burn_in=100)
        plt.hist(samples, bins=100)
        plt.show()

if __name__ == '__main__':
    unittest.main()