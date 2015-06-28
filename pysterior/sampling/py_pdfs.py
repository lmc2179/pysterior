from scipy.stats import multivariate_normal
import numpy as np
from math import log

def mv_normal_exponent(x, mean, cov):
    return -0.5*np.dot(np.dot((x-mean),np.linalg.inv(cov)),(x-mean))

def mv_normal_exponent_gradient(x, mean, cov):
    pass

# x = np.array([0.5,0])
# mean = np.array([0,0])
# cov = np.eye(2,2)
# print(mv_normal_exponent(x, mean, cov))