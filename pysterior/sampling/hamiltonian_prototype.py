# This is a test of the HMCMC method. It will be constructed as a collection of lightweight functions, and then
# refactored into composable middleweight abstractions similar to samplers.py.
import numpy as np
import matplotlib.pyplot as plt
import random
import theano
import theano.tensor as T
import math
from progress.bar import Bar

LOG_ONE = math.log(1.0)

class LeapfrogIntegrator(object):
    def __init__(self, target_energy_gradient):
        self.target_energy_gradient = target_energy_gradient

    def _leapfrog_step(self, value, momentum, step_size):
        half_step_momentum = momentum + (step_size*0.5*self.target_energy_gradient(value))
        step_value = value + (step_size*half_step_momentum)
        step_momentum = half_step_momentum + (step_size*0.5*self.target_energy_gradient(step_value))
        return step_value, step_momentum

    def run_leapfrog(self, current_value, current_momentum, num_steps, step_size):
        value, momentum = current_value, current_momentum
        for i in range(num_steps):
            value,momentum = self._leapfrog_step(value, momentum, step_size)
        return value, momentum

class HamiltonianSampler(object):
    def __init__(self, target_energy):
        self.target_energy = target_energy

    def calculate_acceptance_probability(self, current_value, sampled_momentum, proposed_value,
                                                                      proposed_momentum, target_energy):
        metropolis_num = (target_energy(proposed_value) - (0.5*np.dot(proposed_momentum,proposed_momentum)))
        metropolis_denom = (target_energy(current_value) - (0.5*np.dot(sampled_momentum,sampled_momentum)))
        metropolis_factor = metropolis_num - metropolis_denom
        return min(metropolis_factor, LOG_ONE)

    def accept(self, acceptance_probability):
        return math.log(random.random()) < acceptance_probability

    def run_hamiltonian_sampling(self, initial_value, num_steps, step_size, iterations, burn_in=None):
        b = Bar('Sampling', max=iterations, suffix='%(percent).1f%% - %(eta)ds')
        try:
            dimension = len(initial_value)
        except:
            dimension = 1
        current_value = initial_value
        integrator = LeapfrogIntegrator(self.target_energy.target_log_pdf_gradient)
        samples = []
        for i in range(iterations):
            if dimension > 1:
                sampled_momentum = np.random.multivariate_normal(np.zeros(dimension), np.eye(dimension))
            else:
                sampled_momentum = np.random.normal(0, 1)
            proposed_value, proposed_momentum = integrator.run_leapfrog(current_value, sampled_momentum, num_steps,
                                                                        step_size)
            acceptance_probability = self.calculate_acceptance_probability(current_value, sampled_momentum, proposed_value,
                                                                      proposed_momentum, self.target_energy.target_log_pdf)
            if self.accept(acceptance_probability):
                current_value = proposed_value
            if (burn_in and burn_in < i) or not burn_in:
                samples.append(current_value)
            b.next()
        b.finish()
        return samples

import abc

class AbstractEnergy(object):
    __meta__ = abc.ABCMeta

    @abc.abstractmethod
    def target_log_pdf(self, state):
        "Log energy at the given state."

    @abc.abstractmethod
    def target_log_pdf_gradient(self, state):
        "Gradient of the log energy at a given state."

class GaussianEnergy(AbstractEnergy):
    def __init__(self, mean, covariance):
        x = T.vector('x')
        mu = T.vector('mu')
        inv_cov_matrix = T.matrix('inv_cov_matrix')
        likelihood = -T.dot(T.dot((x-mu).T,inv_cov_matrix),(x-mu))
        self.gaussian_energy = theano.function([x,mu,inv_cov_matrix], likelihood)
        self.gaussian_energy_gradient = theano.function([x, mu, inv_cov_matrix], theano.grad(likelihood, x))
        self.mean = mean
        self.inv_cov = np.linalg.inv(covariance)

    def target_log_pdf(self, X):
        return self.gaussian_energy(X, self.mean, self.inv_cov)

    def target_log_pdf_gradient(self, X):
        return self.gaussian_energy_gradient(X, self.mean, self.inv_cov)

class UnivariateGaussianEnergy(AbstractEnergy):
    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance

    def target_log_pdf(self, X):
        return -((1.0/self.variance)*(X-self.mean)**2)

    def target_log_pdf_gradient(self, X):
        return -((1.0/self.variance)*2*(X - self.mean))

class UnivariateTEnergy(AbstractEnergy):
    def __init__(self, dof):
        self.dof = dof

    def target_log_pdf(self, X):
        return math.log((1+X**2/self.dof)**(-(self.dof+1)/2))

    def target_log_pdf_gradient(self, X):
        return -(((self.dof+1)*X)/(self.dof + X**2))

def test_multivariate_gaussian():
    samples = HamiltonianSampler(GaussianEnergy(np.array([0.0,0.0]), np.array([[1,0],[1,1]]))).run_hamiltonian_sampling(np.array([100.0, 100.0]), 100, 0.05, 5000, burn_in=100)
    plt.plot(*list(zip(*samples)), marker = '.', linewidth=0.0)

def test_univariate_gaussian():
    samples = HamiltonianSampler(UnivariateGaussianEnergy(0.0, 3.4)).run_hamiltonian_sampling(100.0, 100, 0.05, 100000, burn_in=10)
    plt.hist(samples, bins=150, normed=True)

def test_univariate_T():
    samples = HamiltonianSampler(UnivariateTEnergy(5)).run_hamiltonian_sampling(100.0, 100, 0.05, 100000, burn_in=100)
    plt.hist(samples, bins=150, normed=True)

test_univariate_gaussian() #TODO: Move to unit tests
# test_univariate_T()
plt.show()