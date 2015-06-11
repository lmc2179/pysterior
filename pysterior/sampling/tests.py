import unittest
import pyximport
import samplers
import proposal_dist

pyximport.install()
import pdfs

#TODO: Exp direct sampling test
#TODO: Laplace direct sampling test
#TODO: Skew Laplace direct sampling test
#TODO: Kolmogorov-Smirnov testing

#TODO: PosteriorParamter initial test

class GaussianDirectSamplingTest(unittest.TestCase):
    MU, SIGMA = 10, 13.7
    def _get_target_log_pdf(self):
        pdf_closure = lambda x: pdfs.lognormpdf(x, self.MU, self.SIGMA)
        return pdf_closure

    def _get_proposal_distribution(self):
        return proposal_dist.GaussianMetropolisProposal(6.0)

    def test_sampling(self):
        sampler = samplers.MetropolisHastings(self._get_target_log_pdf(),
                                              self._get_proposal_distribution())
        samples = sampler.sample(90000,50000,2, 32.3)
        sample_mean = sum(samples)/len(samples)
        sample_variance = 1.0*sum([(s - sample_mean)**2 for s in samples])/(len(samples)-1)
        self.assertAlmostEqual(sample_mean, self.MU, delta=0.5, msg='Sample mean does not approximate theoretical mean')
        self.assertAlmostEqual(sample_variance, self.SIGMA**2, delta=20.0,
                               msg='Sample mean does not approximate theoretical variance')

class GaussianParameterInference(unittest.TestCase):
    def test_parameter_sampling(self):
        prior_log_pdf = None
        data_log_likelihood = None
        proposal = proposal_dist.SphereGaussianMetropolisProposal(1.0, 2)
        sampler = samplers.ParameterPosteriorSample(prior_log_pdf, data_log_likelihood, proposal)
