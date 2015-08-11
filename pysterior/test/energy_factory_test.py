import unittest
from pysterior import energy
import numpy as np

#TODO: Roll this into a factory
fxn_spec = energy.get_unit_normal_spec()
factory = energy.PartiallyDifferentiableFunctionFactory(fxn_spec)
f, grad = factory.get_partial_diff('X')
f_closure = energy.build_arg_closure(f, {'mu': np.array([0.0, 0.0])}, 'X')
grad_closure = energy.build_arg_closure(grad, {'mu': np.array([0.0, 0.0])}, 'X')
data = [np.array([x1,x2]) for x1,x2 in [(1,1), (-1,-1), (1,-1), (-1, 1)]]
for d in data:
    print(d, f_closure(d), grad_closure(d))
