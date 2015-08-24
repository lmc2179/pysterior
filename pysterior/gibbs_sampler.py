import numpy as np
from pysterior import energy

class GibbsSampler(object):
    def __init__(self, target_variables, total_energy_function, energy_gradients):
        self.target_variables = target_variables
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
        current_state[variable] += 1
        return current_state

    def _build_energy(self, current_state, variable):
        return energy.Energy()

    def _vectorize_state(self, state):
        return np.array([state[v] for v in self.target_variables])