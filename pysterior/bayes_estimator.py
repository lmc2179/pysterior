import numpy as np

class GradientDescent(object):
    def minimize(self, target_function, starting_point, iterations, tol):
        "Takes an instance of AbstractDifferentiableFunction, and runs minimization with unit learning rate."
        LEARNING_RATE = 0.9
        p = starting_point
        for i in range(iterations):
            new_p = p - np.dot(target_function.gradient(p),LEARNING_RATE)
            if abs(new_p-p) < tol:
                return new_p
            else:
                p = new_p
        return p