import unittest
import pyximport
import samplers
import proposal_dist

pyximport.install()
import pdfs

#TODO: Exp test
#TODO: Laplace test

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
        self.assertAlmostEqual(sample_mean, self.MU, delta=0.5, msg='Sample mean does not approximate theoretical mean')