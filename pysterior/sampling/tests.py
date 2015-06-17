import random
import unittest
import pyximport
import samplers
import proposal_dist
from scipy.stats import shapiro, multivariate_normal
import py_pdfs
from matplotlib import pyplot as plt
import numpy as np
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

        def _get_sampler_class(self):
            "Provide a class which implements the a sample() method and takes a target/proposal distribution."
            return samplers.MetropolisHastings

        def _build_sampler(self):
            sampler_cls = self._get_sampler_class()
            return sampler_cls(self._get_target_log_pdf(),  self._get_proposal_distribution())

    class GaussianDirectSamplingTest(OneDimensionalDirectSamplingTest):
        "Test direct sampling from a 1D Gaussian. Provides a target distribution, but still requires a proposal and a sampler."
        MU, SIGMA = 10, 13.7
        def _get_target_log_pdf(self):
            pdf_closure = lambda x: pdfs.lognormpdf(x, self.MU, self.SIGMA)
            return pdf_closure

        def test_sampling(self): #TODO: This is overweight, and will be abstracted properly when we start testing other distributions
            sampler = self._build_sampler()
            samples = sampler.sample(190000,50000,2, -302.3)
            sample_mean = sum(samples)/len(samples)
            sample_variance = 1.0*sum([(s - sample_mean)**2 for s in samples])/(len(samples)-1)
            self.assertAlmostEqual(sample_mean, self.MU, delta=0.5, msg='Sample mean does not approximate theoretical mean')
            self.assertAlmostEqual(sample_variance, self.SIGMA**2, delta=20.0,
                                   msg='Sample mean does not approximate theoretical variance')
            print('{0}, Shapiro test: {1}'.format(self.__class__.__name__, shapiro(samples)))
            # plt.hist(samples, bins=200)
            # plt.show()

class MultivariateNormalDirectSamplingTest(AbstractTestCases.OneDimensionalDirectSamplingTest):
    #TODO: This is overweight, combine it with GaussianDirectSamplingTest to form an abstract class
    #TODO: Add a goodness of fit test
    TRUE_MEAN = np.array([-10.0, 10.0])
    TRUE_COV = np.eye(2,2)*5.6
    def _get_proposal_distribution(self):
        return proposal_dist.GaussianMetropolisProposal(np.eye(2,2)*1.0)

    def _get_target_log_pdf(self):
        # pdf_closure = lambda x: multivariate_normal.logpdf(x, self.TRUE_MEAN, self.TRUE_COV)
        pdf_closure = lambda x: py_pdfs.mv_normal_exponent(x, self.TRUE_MEAN, self.TRUE_COV)
        return pdf_closure

    def test_sampling(self):
        sampler = self._build_sampler()
        samples = sampler.sample(50000,10000,2, [-302.3, 100.0])
        print(samples)
        avg_sample = sum(samples)/len(samples)
        print(avg_sample)
        v1_mean, v2_mean = avg_sample
        v1_true, v2_true = self.TRUE_MEAN
        self.assertAlmostEqual(v1_mean, v1_true, delta=0.1)
        self.assertAlmostEqual(v2_mean, v2_true, delta=0.1)
        # plt.plot(*zip(*samples), linewidth=0.0, marker='.')
        # plt.show()

class MHGaussianDirectSamplingTest(AbstractTestCases.GaussianDirectSamplingTest):
    def _get_proposal_distribution(self):
        return proposal_dist.GaussianMetropolisProposal(6.0)

class DynamicMHGaussianDirectSamplingTest(AbstractTestCases.GaussianDirectSamplingTest):
    def _get_proposal_distribution(self):
        return proposal_dist.GaussianAdaptiveMetropolisProposal(6.0, sampling_period=10000, epsilon=0.01)

@unittest.skip('Skipped GaussianParameterInference until posterior sampling is finished\n')
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