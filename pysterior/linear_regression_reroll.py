import numpy as np
import random
from functools import reduce
from scipy.stats.distributions import norm

def propose(current):
    PROPOSAL_VARIANCE = 0.1
    return np.array([random.gauss(w_i, PROPOSAL_VARIANCE) for w_i in current]) #TODO: This can probably be done faster

def prior(w):
    FIXED_PRECISION = 0.0001
    probabilities = norm.pdf(w, 0, 1.0/FIXED_PRECISION)
    return reduce(lambda x,y:x*y, probabilities, 1.0)

def f(x,w):
    return x.dot(w)

def accept(prob_old, prob_new):
    acceptance_prob = min(1, prob_new/prob_old)
    if random.random() < acceptance_prob:
        return True
    else:
        return False

def get_random(X,y):
    i = random.choice(range(len(X)))
    return X[i], y[i]

def parameter_likelihood(w, true_t, expected_t):
    FIXED_OBSERVATION_NOISE_VARIANCE = 5.0
    return prior(w)*norm.pdf(expected_t, true_t, FIXED_OBSERVATION_NOISE_VARIANCE)

def get_posterior_samples(X, y, iterations, burn_in=None, thinning=None):
    dimension = len(X[0])
    w = np.zeros(dimension)
    samples = []
    for i in range(iterations):
        potential_w = propose(w)
        x,t = get_random(X,y)
        expected_t = f(x,w)
        expected_t_potential = f(x, potential_w)
        prob_w = parameter_likelihood(w, t, expected_t)
        prob_potential_w = parameter_likelihood(potential_w, t, expected_t_potential)
        if accept(prob_w, prob_potential_w):
            w = potential_w
        if burn_in and i > burn_in and thinning and i%thinning == 0:
            samples.append(w)
    print(sum(samples)/len(samples))
    w1,w2 = zip(*samples)
    print(min(w1),max(w1))
    print(np.histogram(w1))
    print(min(w2),max(w2))
    print(np.histogram(w2))

X = np.array([[0,0],[0,1], [1,1], [50,2]])
y = np.array([0,50.5,50.1, 101])
get_posterior_samples(X,y, 50000, burn_in=30000, thinning=2)