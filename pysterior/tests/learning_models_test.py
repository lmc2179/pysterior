import unittest
from pysterior import learning_models, sampler, parametric_functions

class StubSamplingTest(unittest.TestCase):
    def test_stub_sampling(self):
        "Test that the sampling pipeline produces a posterior sample object with the expected mean."
        model = learning_models.RegressionModel(None,
                                        sampler.HamiltonianSamplerStub,
                                        parametric_functions.LinearModel)
        model.sample_parameter_posterior(100000)
        samples = model.get_samples()
        mean = samples.get_mean()
        for dimension_mean in mean:
            self.assertAlmostEqual(dimension_mean, 0.0, delta=1e-2)