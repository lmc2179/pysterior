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

def nuts3(initial_point, epsilon, energy, iterations):
    samples = []
    current_sample = initial_point
    for i in range(iterations):
        momentum = np.random.normal()
        total_energy = energy.eval(current_sample) - (0.5*(momentum*momentum))
        slice_edge = random.uniform(0, math.exp(total_energy))
        forward = back = current_sample
        forward_momentum = back_momentum = momentum
        next_sample = current_sample
        n = 1
        s = 1
        j = 0
        while s == 1:
            direction = random.choice([-1, 1])
            if direction == 1:
                _, _, forward, forward_momentum, candidate_point, candidate_n, candidate_s = build_tree(forward, forward_momentum, slice_edge, direction, j, epsilon, energy)
            else:
                back, back_momentum, _, _, candidate_point, candidate_n, candidate_s = build_tree(back, back_momentum, slice_edge, direction, j, epsilon, energy)
            if candidate_s == 1:
                prob = min(1.0, 1.0*candidate_n/n)
                if random.random() < prob:
                    next_sample = candidate_point
            n = n + candidate_n
            s = candidate_s * I((forward - back) * back_momentum > 0) * I((forward - back) * forward_momentum > 0)
            j += 1
        samples.append(next_sample)
        current_sample = next_sample
    return samples

@functools.lru_cache()
def build_tree(point, momentum, slice_edge, direction, j, epsilon, energy):
    leapfrog = sampler.LeapfrogIntegrator(energy.gradient)
    if j == 0:
        p, r = leapfrog.run_leapfrog(point, momentum, 1, direction*epsilon)
        candidate_n = I(slice_edge < math.exp(energy.eval(p) - (0.5 * r**2)))
        candidate_s = I(energy.eval(p) - (0.5 * r**2) > math.log(slice_edge) - 1000)
        return p, r, p, r, p, candidate_n, candidate_s
    else:
        back, back_momentum, forward, forward_momentum, candidate_point, candidate_n, candidate_s = build_tree(point, momentum, slice_edge, direction, j-1, epsilon, energy)
        if candidate_s:
            if direction == 1:
                _, _, forward, forward_momentum, candidate_point_2, candidate_n_2, candidate_s_2 = build_tree(forward, forward_momentum, slice_edge, direction, j-1, epsilon, energy)
            else:
                back, back_momentum, _, _, candidate_point_2, candidate_n_2, candidate_s_2 = build_tree(back, back_momentum, slice_edge, direction, j-1, epsilon, energy)
            if candidate_n_2 > 0 and random.random() < (candidate_n_2 / (candidate_n_2 + candidate_n)):
                candidate_point = candidate_point_2
            candidate_s = candidate_s_2 * I((forward - back) * back_momentum > 0) * I((forward - back) * forward_momentum > 0)
            candidate_n = candidate_n + candidate_n_2
        return back, back_momentum, forward, forward_momentum, candidate_point, candidate_n, candidate_s

def I(statement):
    if statement == True:
        return 1
    else:
        return 0

energy = GaussianEnergyClosure(0.0, 5.0)
samples = nuts3(50.0, 0.07, energy, 5000)
print(samples)
plt.hist(samples, bins=100, normed=True)
plt.show()