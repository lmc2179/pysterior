class GibbsSampler(object):
    def __init__(self, target_variables, total_energy_function, energy_gradients):
        self.target_variables = target_variables
        self.total_energy_function = total_energy_function
        self.energy_gradients = energy_gradients

    def run_sampling(self, initial_state, iterations):
        pass