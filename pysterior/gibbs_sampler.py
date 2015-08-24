import numpy as np
from pysterior import energy, sampler

class GibbsSampler(object):
    def __init__(self, target_variables, total_energy_function, energy_gradients):
        self.target_variables = target_variables
        self.epsilon = {v:None for v in target_variables}
        self.total_energy_function = total_energy_function
        self.energy_gradients = energy_gradients

    def run_sampling(self, initial_state, iterations):
        samples = []
        state = initial_state
        for i in range(iterations):
            state = self._get_next_state(state)
            vectorized_state = self._vectorize_state(state)
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
        return energy.Energy() #TODO: Here is where it gets ugly

    def _sample_variable_value(self, current_variable_value, E, variable):
        nuts = sampler.NUTS()
        e = self.epsilon[variable]
        if not e:
            e = sampler.RobbinsMonroEpsilonEstimator().estimate_epsilon(E, current_variable_value, 10)
            self.epsilon[variable] = e
        sample = nuts.nuts_with_fixed_epsilon(current_variable_value, E, e, iterations=1)
        return sample[0]


    def _vectorize_state(self, state):
        return np.array([state[v] for v in self.target_variables])