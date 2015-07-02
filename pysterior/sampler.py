import data_model
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

class AbstractHamiltonianSampler(object):
    def __init__(self, target_energy):
        self.target_energy = target_energy

    #TODO: Acceptance in abstract class

    def sample(self, iterations, burn_in=0, thinning=1):
        raise NotImplementedError

class HamiltonianSamplerStub(AbstractHamiltonianSampler):
    def sample(self, iterations, burn_in=0, thinning=1):
        samples = data_model.PosteriorSample()
        for i in range(iterations):
            samples.add_sample(np.random.multivariate_normal(np.zeros(3), np.eye(3,3)))
        return samples