import unittest
from pysterior import parametric_functions
import numpy as np

class LinearModelTest(unittest.TestCase):
    def test_linear_model(self):
        f = parametric_functions.LinearModel()
        print(f(np.ones(3), np.array([0,1,2])))