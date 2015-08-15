from pysterior import energy, sampler
import copy
import random

class GibbsDirectSampler(object):
    def __init__(self, fxn_spec, target_variables, bound_parameters):
        self.fxn_spec = fxn_spec
        self.bound_parameters = bound_parameters
        self.target_variables = target_variables
        self.target_energy_function, self.gradients = self._build_gradients(fxn_spec, target_variables)
        self.sampler = sampler.NUTS()

    def _build_gradients(self, fxn_spec, target_variables):
        pdiff = energy.PartiallyDifferentiableFunctionFactory(fxn_spec)
        results = [pdiff.get_partial_diff(v) for v in target_variables]
        return results[0][0], [r[1] for r in results] #Clumsy

    def run_sampling(self, iterations, initial_point, burn_in=10):
        X = initial_point
        samples = []
        for i in range(iterations+burn_in):
            next_X = copy.deepcopy(X)
            for j, var in enumerate(self.target_variables):
                next_X[j] = self._get_next_sample(var, next_X, j)
            if i >= burn_in:
                samples.append(next_X)
            X = next_X
        return samples

    def _get_next_sample(self, var_name, current_state, var_index):
        #TODO: Construct the energy function, run a single sample
        E = self._build_energy_function(var_name, current_state, var_index)
        sample = self.sampler.nuts_with_initial_epsilon(current_state[var_index], E, 1, 2)[0]
        return sample

    def _build_energy_function(self, var_name, current_state, var_index):
        #TODO: construct the closed function and gradient, and add to Energy tuple
        closed_arguments = {}
        closed_arguments.update({v: current_state[i] for i,v in enumerate(self.target_variables) if i != var_index})
        closed_arguments.update(self.bound_parameters)
        closed_f = energy.build_arg_closure(self.target_energy_function,
                                            closed_arguments,
                                            var_name)
        closed_grad = energy.build_arg_closure(self.gradients[var_index],
                                            closed_arguments,
                                            var_name)
        return energy.Energy(eval=closed_f,
                            gradient=closed_grad)