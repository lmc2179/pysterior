import numpy as np
import random

#TODO: Armijo-Goldstein
class StochasticBatchGradientDescent(object):
    def minimize(self, target_functions, starting_point, iterations, batch_size=1):
        "Takes an instance of AbstractDifferentiableFunction, and runs minimization with basic adaptive learning rate."
        INITIAL_LEARNING_RATE = 1.0
        convergence_parameter = 0.6
        p = starting_point
        for i in range(iterations):
            learning_rate =(INITIAL_LEARNING_RATE) / ((1 + INITIAL_LEARNING_RATE*convergence_parameter*(i+1))**(0.75))
            target_function_batch = random.sample(target_functions, batch_size)
            p = self._get_next_point_batch(target_function_batch, p, learning_rate)
        return p

    def _get_next_point(self, target_function, p, learning_rate):
        next_point = p - np.dot(target_function.gradient(p), learning_rate)
        return next_point

    def _get_next_point_batch(self, target_function_batch, p, learning_rate):
        target_function_folded = lambda x: sum([function.gradient(x) for function in target_function_batch])
        next_point = p - np.dot(target_function_folded(p), learning_rate)
        return next_point