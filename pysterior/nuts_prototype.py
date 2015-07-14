import abstract_differentiable_function
import theano.tensor as T
import numpy as np
import random
import math
import functools
import sampler
import matplotlib.pyplot as plt

class GaussianEnergy(abstract_differentiable_function.AbstractDifferentiableFunction):
    def _get_variables(self):
        X = T.scalar('x')
        mu = T.scalar('mu')
        sigma = T.scalar('sigma')
        y = -(X - mu)**2 * (1.0/sigma)
        return [X], [mu, sigma], y

class GaussianEnergyClosure(object):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.delegate_energy = GaussianEnergy()

    def eval(self, x):
        return self.delegate_energy.eval(x, self.mu, self.sigma)

    def gradient(self, x):
        return self.delegate_energy.gradient(x, self.mu, self.sigma)

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

class Nuts3(object):
    def _select_heuristic_epsilon(self, energy, initial_point):
        epsilon = 1
        momentum = self._sample_momentum()
        leapfrog = LeapfrogIntegrator(energy.gradient)
        next_point, next_momentum = leapfrog.run_leapfrog(initial_point, momentum, 1, epsilon)
        a = 2*I(self._get_log_probability(energy, initial_point, momentum) - self._get_log_probability(energy, next_point, next_momentum) > math.log(0.5)) - 1
        while(a*((self._get_log_probability(energy, initial_point, momentum) - self._get_log_probability(energy, next_point, next_momentum))) > math.log(2**-a)):
            epsilon = (2**a) * epsilon
            next_point, next_momentum = leapfrog.run_leapfrog(initial_point, momentum, 1, epsilon)
        return epsilon

    def nuts3(self, initial_point, energy, iterations, burn_in=0):
        epsilon = self._select_heuristic_epsilon(energy, initial_point)
        print('Selected epsilon = ', epsilon)
        samples = []
        current_sample = initial_point
        for i in range(iterations+burn_in):
            momentum = self._sample_momentum()
            slice_edge = random.uniform(0, self._get_probability(energy, current_sample, momentum))
            forward = back = current_sample
            forward_momentum = back_momentum = momentum
            next_sample = current_sample
            n = 1
            no_u_turn = True
            j = 0
            while no_u_turn:
                direction = random.choice([-1, 1])
                if direction == 1:
                    _, _, forward, forward_momentum, candidate_point, candidate_n, candidate_no_u_turn = self._build_tree(forward, forward_momentum, slice_edge, direction, j, epsilon, energy)
                else:
                    back, back_momentum, _, _, candidate_point, candidate_n, candidate_no_u_turn = self._build_tree(back, back_momentum, slice_edge, direction, j, epsilon, energy)
                if candidate_no_u_turn:
                    prob = min(1.0, 1.0*candidate_n/n)
                    if random.random() < prob:
                        next_sample = candidate_point
                n = n + candidate_n
                no_u_turn = candidate_no_u_turn and ((forward - back) * back_momentum > 0) and ((forward - back) * forward_momentum > 0)
                j += 1
            if i > burn_in:
                samples.append(next_sample)
            current_sample = next_sample
        return samples

    @functools.lru_cache()
    def _build_tree(self, point, momentum, slice_edge, direction, j, epsilon, energy):
        leapfrog = LeapfrogIntegrator(energy.gradient)
        if j == 0:
            p, r = leapfrog.run_leapfrog(point, momentum, 1, direction*epsilon)
            if slice_edge > 0: #TODO: This is an ugly hack; find a numerically stable solution or hide it when we refactor
                candidate_n = I(slice_edge < self._get_probability(energy, p, r))
                candidate_no_u_turn = (self._get_probability(energy, p, r) > math.log(slice_edge) - 1000)
            else:
                candidate_n = 1
                candidate_no_u_turn = True
            return p, r, p, r, p, candidate_n, candidate_no_u_turn
        else:
            back, back_momentum, forward, forward_momentum, candidate_point, candidate_n, candidate_no_u_turn = self._build_tree(point, momentum, slice_edge, direction, j-1, epsilon, energy)
            if candidate_no_u_turn:
                if direction == 1:
                    _, _, forward, forward_momentum, candidate_point_2, candidate_n_2, candidate_2_no_u_turn = self._build_tree(forward, forward_momentum, slice_edge, direction, j-1, epsilon, energy)
                else:
                    back, back_momentum, _, _, candidate_point_2, candidate_n_2, candidate_2_no_u_turn = self._build_tree(back, back_momentum, slice_edge, direction, j-1, epsilon, energy)
                if candidate_n_2 > 0 and random.random() < (candidate_n_2 / (candidate_n_2 + candidate_n)):
                    candidate_point = candidate_point_2
                candidate_no_u_turn = candidate_2_no_u_turn and ((forward - back) * back_momentum > 0) and ((forward - back) * forward_momentum > 0)
                candidate_n = candidate_n + candidate_n_2
            return back, back_momentum, forward, forward_momentum, candidate_point, candidate_n, candidate_no_u_turn

    def _sample_momentum(self):
        return np.random.normal()

    def _get_probability(self, energy, p, r):
        return math.exp(energy.eval(p) - (0.5 * r ** 2))

    def _get_log_probability(self, energy, p, r):
        return energy.eval(p) - (0.5 * r ** 2)


def I(statement):
    if statement == True:
        return 1
    else:
        return 0

energy = GaussianEnergyClosure(0.0, 5.0)
samples = Nuts3().nuts3(100.0, energy, 4000, burn_in=100)
# print(samples)
# print(shapiro(samples))
plt.hist(samples, bins=100, normed=True)
plt.show()