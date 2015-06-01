from linear_regression import BayesianLinearRegression
import random
import numpy as np
import unittest
from matplotlib import pyplot as plt

class RegressionTest(unittest.TestCase):
    def test_linear_regression(self):
        TRUE_WEIGHTS = np.array([-13.5,50.0])
        X = np.array([[random.randint(-100,100), random.randint(-100,100)] for i in range(20)])
        y = np.array([TRUE_WEIGHTS.dot(x) for x in X])
        model = BayesianLinearRegression()
        model.fit_sample(X,y, 50000, burn_in=25000, thinning=2)
        sample_means = model.get_parameter_sample_mean()
        ERROR = 0.01
        for true_w, mean_w in zip(TRUE_WEIGHTS, sample_means):
            self.assertAlmostEqual(true_w, mean_w, delta=ERROR)

@unittest.skip('Demos are not run as part of unit testing')
class TwoDimensionalLinearRegressionDemo(unittest.TestCase):
    def test_histogram(self):
        TRUE_WEIGHTS = np.array([-13.5,50.0])
        X = np.array([[random.randint(-100,100), random.randint(-100,100)] for i in range(20)])
        y = np.array([TRUE_WEIGHTS.dot(x) for x in X])
        model = BayesianLinearRegression()
        model.fit_sample(X,y, 50000, burn_in=25000, thinning=2)
        parameter_samples = model.get_parameter_samples()
        fig = plt.figure()
        for i, parameter_sample_set in enumerate(parameter_samples):
            subplot = fig.add_subplot(int('{0}{1}{2}'.format(len(parameter_samples),1,i+1)))
            self._graph_posterior_histogram(subplot, parameter_sample_set)
        plt.show()

    def _graph_posterior_histogram(self, subplot, parameter_sample_set):
        subplot.hist(parameter_sample_set, bins=300)

    def test_bimodal_histogram(self): #TODO: This does not work with symmetric metropolis
        TRUE_WEIGHTS1 = np.array([-13.5,50.0])
        TRUE_WEIGHTS2 = np.array([-13.5,-50.0])
        data = [[random.randint(-100,100), random.randint(-100,100)] for i in range(20)]
        import copy
        X = np.array(copy.deepcopy(data) + copy.deepcopy(data))
        y = np.array([TRUE_WEIGHTS1.dot(x) for x in data] + [TRUE_WEIGHTS2.dot(x) for x in data])
        model = BayesianLinearRegression()
        model.fit_sample(X,y, 100000, burn_in=70000, thinning=2)
        parameter_samples = model.get_parameter_samples()
        fig = plt.figure()
        for i, parameter_sample_set in enumerate(parameter_samples):
            subplot = fig.add_subplot(int('{0}{1}{2}'.format(len(parameter_samples),1,i+1)))
            self._graph_posterior_histogram(subplot, parameter_sample_set)
        plt.show()