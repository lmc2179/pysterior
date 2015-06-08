import unittest
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import kstest, mstats
from scipy.stats.distributions import laplace, norm
from math import exp, log
import samplers

class AbstractSamplerTest(unittest.TestCase):
    def _get_samples(self):
        raise NotImplementedError

    def _get_true_cdf(self):
        raise NotImplementedError

    def _get_alpha(self):
        raise NotImplementedError

    def test_kolmogorov_smirnov(self):
        samples = self._get_samples()
        cdf = self._get_true_cdf()
        alpha = self._get_alpha()
        D,p = kstest(samples, cdf)
        self.assertGreater(p, alpha, 'p={0}, alpha={1}'.format(p, alpha))


class MetropolisSamplerTest(unittest.TestCase):
    TRUE_MU, TRUE_SIGMA = 10.0, 3.4

    def _get_samples(self):
        pdf_closure = lambda x: norm.logpdf(x, self.TRUE_MU, self.TRUE_SIGMA)
        sampler = samplers.GaussianMetropolis1D(1.0, pdf_closure)
        samples = sampler.sample(100000, 50000, thinning=20)
        return samples

    def _get_alpha(self):
        return 0.05

    def test_normality(self):
        samples = self._get_samples()
        alpha = self._get_alpha()
        D, p = mstats.normaltest(samples) #TODO: This fails sometimes
        self.assertGreater(p, alpha, 'p={0}, alpha={1}'.format(p, alpha))


def skew_norm_pdf(x, location, scale, shape):
    from scipy.stats.distributions import norm
    from math import log
    if not shape:
        y = (x-location)/scale
    else:
        y = (-1.0/shape)*log(1 - ((shape*(x-location))/(scale)))
    return norm.pdf(y, 0, 1)/(scale - shape*(x - location))

def skew_laplace_pdf(x, location, left_scale, right_scale):
    partition = 1 / (left_scale + right_scale)
    if x <= location:
        return partition * exp((x - location) / left_scale)
    if x > location:
        return partition * exp((location - x) / right_scale)

def skew_laplace_log_pdf(x, location, left_scale, right_scale):
    partition = 1 / (left_scale + right_scale)
    if x <= location:
        return log(partition) + ((x - location) / left_scale)
    if x > location:
        return log(partition) + ((location - x) / right_scale)


class LaplacianMetropolisTest(AbstractSamplerTest):
    TRUE_LOC, TRUE_SCALE = 0, 1.0

    def _get_samples(self):
        pdf_closure = lambda x: laplace.logpdf(x, self.TRUE_LOC, self.TRUE_SCALE)
        sampler = samplers.GaussianMetropolis1D(1.0, pdf_closure)
        samples = sampler.sample(100000, 50000, thinning=10)
        return samples

    def _get_alpha(self):
        return 1e-3

    def _get_true_cdf(self):
        cdf_closure = lambda x: laplace.cdf(x, self.TRUE_LOC, self.TRUE_SCALE)
        return cdf_closure

    def draw_laplace_samples_pdf(self):
        samples = self._get_samples()
        plt.hist(samples, bins=200)
        plt.show()

    def draw_skew_pdf(self):
        X = np.linspace(-4, 4, 110)
        y = [laplace.pdf(x, self.TRUE_LOC, self.TRUE_SCALE) for x in X]
        plt.plot(X , y)
        plt.show()


class SkewLaplacianMetropolisTest(unittest.TestCase): #TODO: Convert to K-S test
    TRUE_LOC, TRUE_LEFT_SCALE, TRUE_RIGHT_SCALE = 0, 1.0, 4.0

    def _get_samples(self):
        pdf_closure = lambda x: skew_laplace_log_pdf(x, self.TRUE_LOC, self.TRUE_LEFT_SCALE, self.TRUE_RIGHT_SCALE)
        sampler = samplers.GaussianMetropolis1D(1.0, pdf_closure)
        samples = sampler.sample(100000, 50000, thinning=1)
        return samples

    def test_skew_pdf(self):
        samples = self._get_samples()
        plt.hist(samples, bins=150)
        plt.show()

    def draw_skew_pdf(self):
        X = np.linspace(-4, 4, 110)
        y = [skew_laplace_log_pdf(x, self.TRUE_LOC, self.TRUE_LEFT_SCALE, self.TRUE_RIGHT_SCALE) for x in X]
        plt.plot(X , y)
        plt.show()
