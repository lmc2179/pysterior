import copy

import numpy as np

from archive import sampler, energy


class GibbsSampler(object):
    RM_ITERATIONS = 10
    def __init__(self, target_variables, total_energy_function, energy_gradients):
        self.target_variables = target_variables
        self.epsilon = {v:None for v in target_variables}
        self.total_energy_function = total_energy_function
        self.energy_gradients = {v:g for v,g in zip(target_variables, energy_gradients)}

    def run_sampling(self, initial_state, iterations, burn_in=0):
        samples = []
        state = initial_state
        for i in range(iterations+burn_in):
            state = self._get_next_state(state)
            vectorized_state = self._vectorize_state(state)
            if i >= burn_in:
                samples.append(vectorized_state)
        return samples

    def _get_next_state(self, current_state):
        for variable in self.target_variables:
            current_state = self._mutate_state(current_state, variable)
        return current_state

    def _mutate_state(self, current_state, variable):
        E = self._build_energy(current_state, variable)
        current_variable_value = current_state[variable]
        new_variable_value = self._sample_variable_value(current_variable_value, E, variable)
        current_state[variable] = new_variable_value
        return current_state

    def _build_energy(self, current_state, variable):
        closed_state = copy.copy(current_state)
        del closed_state[variable]
        f = self.build_arg_closure(self.total_energy_function, closed_state, variable)
        grad = self.build_arg_closure(self.energy_gradients[variable],
                                 closed_state,
                                 variable)
        return energy.Energy(eval=f, gradient=grad)


    def build_arg_closure(self, f, bound_kwargs, unbound_arg_name):
        def partial_fxn(arg):
            full_kwargs = {}
            full_kwargs.update(bound_kwargs)
            full_kwargs.update({unbound_arg_name: arg})
            return f(**full_kwargs)
        return partial_fxn


    def _sample_variable_value(self, current_variable_value, E, variable):
        nuts = sampler.NUTS()
        e = self.epsilon[variable]
        if not e:
            e = sampler.RobbinsMonroEpsilonEstimator().estimate_epsilon(E, current_variable_value, self.RM_ITERATIONS)
            self.epsilon[variable] = e
        sample = nuts.nuts_with_fixed_epsilon(current_variable_value, E, e, iterations=1)
        return sample[0]


    def _vectorize_state(self, state):
        return np.array([state[v] for v in self.target_variables])