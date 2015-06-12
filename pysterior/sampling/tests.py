import unittest
import pyximport
import samplers
import proposal_dist
pyximport.install()
import abc
import pdfs

#TODO: Exp direct sampling test
#TODO: Laplace direct sampling test
#TODO: Skew Laplace direct sampling test
#TODO: Kolmogorov-Smirnov testing
#TODO: Multivariate Gaussian direct sampling

#TODO: PosteriorParamter initial test

class AbstractTestCases(object):
    "A wrapper around the abstract test case classes, so that unittest.main doesn't pick them up."
    class OneDimensionalDirectSamplingTest(unittest.TestCase):
        __meta__ = abc.ABCMeta

        @abc.abstractmethod
        def _get_target_log_pdf(self):
            "Returns a function, which is the log_pdf of the target distribution."

        @abc.abstractmethod
        def _get_proposal_distribution(self):
            "Returns a proposal distribution object."

        @abc.abstractmethod
        def _get_sampler_class(self):
            "Provide a class which implements the a sample() method and takes a target/proposal distribution."

    class GaussianDirectSamplingTest(OneDimensionalDirectSamplingTest):
        "Test direct sampling from a 1D Gaussian. Provides a target distribution, but still requires a proposal and a sampler."
        MU, SIGMA = 10, 13.7
        def _get_target_log_pdf(self):
            pdf_closure = lambda x: pdfs.lognormpdf(x, self.MU, self.SIGMA)
            return pdf_closure

        def _build_sampler(self):
            sampler_cls = self._get_sampler_class()
            return sampler_cls(self._get_target_log_pdf(),  self._get_proposal_distribution())

        def _get_sampler_class(self):
            return samplers.MetropolisHastings

        def test_sampling(self): #TODO: This is overweight, and will be abstracted properly when we start testing other distributions
            sampler = self._build_sampler()
            samples = sampler.sample(90000,50000,2, 32.3)
            sample_mean = sum(samples)/len(samples)
            sample_variance = 1.0*sum([(s - sample_mean)**2 for s in samples])/(len(samples)-1)
            self.assertAlmostEqual(sample_mean, self.MU, delta=0.5, msg='Sample mean does not approximate theoretical mean')
            self.assertAlmostEqual(sample_variance, self.SIGMA**2, delta=20.0,
                                   msg='Sample mean does not approximate theoretical variance')

class MHGaussianDirectSamplingTest(AbstractTestCases.GaussianDirectSamplingTest):
    def _get_proposal_distribution(self):
        return proposal_dist.GaussianMetropolisProposal(6.0)

class DynamicMHGaussianDirectSamplingTest(AbstractTestCases.GaussianDirectSamplingTest):
    def _get_proposal_distribution(self):
        return proposal_dist.GaussianAdaptiveMetropolisProposal(6.0)

@unittest.skip('Skipped GaussianParameterInference until posterior sampling is finished')
class GaussianParameterInference(unittest.TestCase):
    def get_prior(self):
        HUGE_VARIANCE = 10000
        flat_normal_closure = lambda x: pdfs.lognormpdf(x, 0, HUGE_VARIANCE)
        return flat_normal_closure

    def _get_data_log_likelihood(self, TRUE_MU, TRUE_SIGMA):
        data_point_likelihood_closure = lambda x: pdfs.lognormpdf(x, TRUE_MU, TRUE_SIGMA)
        return data_point_likelihood_closure

    def test_parameter_sampling(self):
        TRUE_MU, TRUE_SIGMA = 4.0, 2.5
        prior_log_pdf = self.get_prior()
        data_log_likelihood = self._get_data_log_likelihood(TRUE_MU, TRUE_SIGMA)
        proposal = proposal_dist.SphereGaussianMetropolisProposal(1.0, 2)
        sampler = samplers.ParameterPosteriorSample(prior_log_pdf, data_log_likelihood, proposal)
