# This is a pilot of encapsulating pymc3 models
import pymc3
import numpy as np
from matplotlib import pyplot as plt

np.random.seed(123)

# No. 1: Linear Regression
class BayesianLinearRegression(object):
    def sample(self, X, y, iterations):
        model = self._build_model(X, y)
        with model:
            map_estimate = pymc3.find_MAP(model=model)
            step = pymc3.NUTS(scaling=map_estimate)
            trace = pymc3.sample(iterations, step, start=map_estimate)
        return trace

    def _build_model(self, X, y):
        lr_model = pymc3.Model()

        with lr_model:
            alpha = pymc3.Normal(name='alpha', mu=0, sd=20)
            beta = pymc3.Normal(name='beta', mu=0, sd=10, shape=2)
            sigma = pymc3.HalfNormal(name='sigma', sd=1)
            X = pymc3.Normal(name='X', mu=1, sd=2, observed=X)
            mu = alpha + beta.dot(X.T)
            Y_obs = pymc3.Normal(name='Y_obs', mu=mu, sd=sigma, observed=y)

        return lr_model

TRUE_ALPHA, TRUE_SIGMA = 1, 1
TRUE_BETA = [1, 2.5]
size = 100
X1 = np.linspace(0, 1, size)
X2 = np.linspace(0,.2, size)
Y = TRUE_ALPHA + TRUE_BETA[0]*X1 + TRUE_BETA[1]*X2 + np.random.randn(size)*TRUE_SIGMA

X = np.array(list(zip(X1, X2)))

samples = BayesianLinearRegression().sample(X, Y, 5000)
print(samples)
plt.hist(samples['alpha'], bins=100)
plt.show()


