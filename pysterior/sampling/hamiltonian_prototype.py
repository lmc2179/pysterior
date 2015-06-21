# This is a test of the HMCMC method. It will be constructed as a collection of lightweight functions, and then
# refactored into composable middleweight abstractions similar to samplers.py.
import numpy as np
import matplotlib.pyplot as plt
import random
import theano
import theano.tensor as T

x = T.vector('x')
mu = T.vector('mu')
inv_cov_matrix = T.matrix('inv_cov_matrix')
likelihood = T.dot(T.dot((x-mu).T,inv_cov_matrix),(x-mu))
gaussian_energy = theano.function([x,mu,inv_cov_matrix], likelihood)
gaussian_energy_gradient = theano.function([x, mu, inv_cov_matrix], theano.grad(likelihood, x))

#TODO: Figure out how to encapsulate theano variables and functions correctly
TRUE_MEAN, TRUE_COV = np.array([-100,108]), np.eye(2,2)*3.4
INV_COV = np.linalg.inv(TRUE_COV)
def gaussian_log_pdf(X):
    return gaussian_energy(X, TRUE_MEAN, INV_COV)

def gaussian_log_gradient(X):
    return gaussian_energy_gradient(X, TRUE_MEAN, INV_COV)

def calculate_acceptance_probability(current_value, sampled_momentum, proposed_value,
                                                                  proposed_momentum, target_energy):
    metropolis_factor = None #TODO: Add me
    return min(metropolis_factor, 1)

def accept(acceptance_probability):
    return random.random() < acceptance_probability

def leapfrog_step(value, momentum, step_size, energy_gradient):
    half_step_momentum = momentum + (step_size*0.5*energy_gradient(value))
    step_value = value + (step_size*half_step_momentum)
    step_momentum = half_step_momentum + (step_size*0.5*energy_gradient(step_value))
    return step_value, step_momentum

def run_leapfrog(current_value, current_momentum, num_steps, step_size, target_energy_gradient):
    value, momentum = current_value, current_momentum
    for i in range(num_steps):
        value,momentum = leapfrog_step(value, momentum, step_size, target_energy_gradient)
    return value, momentum

def run_hamiltonian_sampling(initial_value, num_steps, step_size, target_energy, target_energy_gradient, iterations):
    dimension = len(initial_value)
    current_value = initial_value
    samples = []
    for i in iterations:
        sampled_momentum = np.random.multivariate_normal(current_value, np.eye(dimension))
        proposed_value, proposed_momentum = run_leapfrog(current_value, sampled_momentum, num_steps, step_size,
                                                         target_energy_gradient)
        acceptance_probability = calculate_acceptance_probability(current_value, sampled_momentum, proposed_value,
                                                                  proposed_momentum, target_energy)
        if accept(acceptance_probability):
            samples.append(proposed_value)
    return samples

samples = run_hamiltonian_sampling(np.array([100.0, -150.0]), 10, 0.01, gaussian_log_pdf, gaussian_log_gradient, 1000)
plt.plot(*samples)
plt.show()

