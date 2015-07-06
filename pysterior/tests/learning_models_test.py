import unittest
from pysterior import learning_models, sampler, parametric_functions
import numpy as np
from scipy.stats import shapiro

class StubSamplingTest(unittest.TestCase):
    def _construct_stub_model(self):
        """
        Construct a stub learning model and return it.

        The model will have samples from a 3D gaussian, and uses a linear model (dot product) as its regression
        function.
        """
        model = learning_models.RegressionModel(None,
                                                sampler.HamiltonianSamplerStub,
                                                parametric_functions.LinearModel())
        model.sample_parameter_posterior(100000)
        return model

    def test_stub_sampling(self):
        "Test that the sampling pipeline produces a posterior sample object with the expected mean."
        model = self._construct_stub_model()
        samples = model.get_samples()
        mean = samples.get_mean()
        for dimension_mean in mean:
            self.assertAlmostEqual(dimension_mean, 0.0, delta=1e-2)

    def test_point_estimate(self):
        model = self._construct_stub_model()
        X = [np.random.multivariate_normal(np.zeros(3), np.eye(3,3)) for i in range(100)]
        predictions = model.predict_point_estimate(X)
        for prediction in predictions:
            self.assertAlmostEqual(prediction, 0.0, delta=1e-1)

    def test_predictive_posterior(self):
        x = np.array([5,5,5])
        model = self._construct_stub_model()
        prediction_samples = model.sample_predictive_posterior(x)
        w,p = shapiro(prediction_samples)
        self.assertGreater(p, 0.75)
