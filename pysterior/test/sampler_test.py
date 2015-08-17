import theano.tensor as T
from theano import function, grad
from pysterior.sampler import NUTS
from pysterior import energy
import matplotlib.pyplot as plt
import unittest

#TODO: Make these automated tests

class TestUnivariateDistributions(unittest.TestCase):
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
        samples = sampler.nuts_with_initial_epsilon(0.0, E, iterations=10000, burn_in=10)
        plt.hist(samples, bins=100)
        plt.show()

    def test_laplace(self):
        TRUE_MEAN = -10.0
        TRUE_SCALE = 3.7
        sampler = NUTS()
        X = T.scalar('X')
        output = -1.0 * (abs(X - TRUE_MEAN) / TRUE_SCALE)
        f = function([X], output, allow_input_downcast=True)
        grad_f = function([X],
                          grad(output, X),
                          allow_input_downcast=True)
        E = energy.Energy(eval=f,
                          gradient=grad_f)
        samples = sampler.nuts_with_initial_epsilon(0.0, E, iterations=10000, burn_in=10)
        plt.hist(samples, bins=100)
        plt.show()

if __name__ == '__main__':
    unittest.main()