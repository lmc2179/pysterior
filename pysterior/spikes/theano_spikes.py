import theano.tensor as T
from theano import dot, function, map, shared
from math import exp, e
import numpy as np

# No. 1: Using map
x = T.vector('x')
W = T.matrix('W')
f = lambda a, b: e**dot(a, b)
results, updates = map(f, sequences=[W], non_sequences=x)

f_np = lambda a, b: e**np.dot(a, b)
v = np.array([0,1,2])
print(f_np(v, v))

calculate_results = function([W, x], outputs=results, allow_input_downcast=True)
test_W = np.array([[0,1,2],[0,1,2]])
test_x = np.array([0,1,2])
print(calculate_results(test_W, test_x))

# No. 2: The softmax function
x = T.vector('x')
y = T.nnet.softmax(x)
smax = function([x], y, allow_input_downcast=True)
print(smax(np.array([0,1,2])))
elements = [0,1,2]
total = sum([exp(el) for el in elements])
print([exp(el)/total for el in elements])
