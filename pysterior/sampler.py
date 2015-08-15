import random
import math
import numpy as np

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

class NUTS(object):
    MAX_EPSILON = 1024.0
    DEFAULT_EPSILON = 1.0
    def _select_heuristic_epsilon(self, energy, initial_point):
        epsilon = 2**(-10) #TODO: Epsilon keeps exploding, added bandage
        momentum = self._sample_momentum(self._get_dimension(initial_point))
        leapfrog = LeapfrogIntegrator(energy.gradient)
        next_point, next_momentum = leapfrog.run_leapfrog(initial_point, momentum, 1, epsilon)
        a = 2*self.I(self._get_log_probability(energy, initial_point, momentum) - self._get_log_probability(energy, next_point, next_momentum) > math.log(0.5)) - 1
        while(a*((self._get_log_probability(energy, initial_point, momentum) - self._get_log_probability(energy, next_point, next_momentum))) > math.log(2**-a)):
            if epsilon > self.MAX_EPSILON:
                return self.DEFAULT_EPSILON
            epsilon = (2**a) * epsilon
            next_point, next_momentum = leapfrog.run_leapfrog(initial_point, momentum, 1, epsilon)
        return epsilon

    def nuts_with_initial_epsilon(self, initial_point, energy, iterations, burn_in=0):
        epsilon = self._select_heuristic_epsilon(energy, initial_point)
        dimension = self._get_dimension(initial_point)
        print('Selected epsilon = ', epsilon)
        samples = []
        current_sample = initial_point
        for i in range(iterations+burn_in):
            momentum = self._sample_momentum(dimension)
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
                no_u_turn = candidate_no_u_turn and (np.dot((forward - back) , back_momentum) > 0) and (np.dot((forward - back) , forward_momentum) > 0)
                j += 1
            if i >= burn_in:
                samples.append(next_sample)
            current_sample = next_sample
        return samples

    def _build_tree(self, point, momentum, slice_edge, direction, j, epsilon, energy):
        leapfrog = LeapfrogIntegrator(energy.gradient)
        if j == 0:
            p, r = leapfrog.run_leapfrog(point, momentum, 1, direction*epsilon)
            if slice_edge > 0: #TODO: This is an ugly hack; find a numerically stable solution or hide it when we refactor
                candidate_n = self.I(slice_edge < self._get_probability(energy, p, r))
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
                candidate_no_u_turn = candidate_2_no_u_turn and (np.dot((forward - back) , back_momentum) > 0) and (np.dot((forward - back) , forward_momentum) > 0)
                candidate_n = candidate_n + candidate_n_2
            return back, back_momentum, forward, forward_momentum, candidate_point, candidate_n, candidate_no_u_turn

    def _sample_momentum(self, dimension):
        if dimension == 0:
            return np.random.normal()
        else:
            return np.random.normal(size=dimension)

    def _get_dimension(self, p):
        try: #TODO: Remove try/except
            return len(p)
        except TypeError:
            return 0

    def _get_probability(self, energy, p, r):
        return math.exp(energy.eval(p) - (0.5 * np.dot(r,r)))

    def _get_log_probability(self, energy, p, r):
        return energy.eval(p) - (0.5 * np.dot(r,r))

    def I(self, statement):
        if statement == True:
            return 1
        else:
            return 0