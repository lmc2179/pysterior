# This is a pilot of encapsulating pymc3 models
import pymc3
import numpy as np

np.random.seed(123)

# No. 1: Linear Regression
def build_lr_model(X, Y):
    lr_model = pymc3.Model()

    with lr_model:
        alpha = pymc3.Normal(name='alpha', mu=0, sd=20)
        beta = pymc3.Normal(name='beta', mu=0, sd=10, shape=2)
        sigma = pymc3.HalfNormal(name='sigma', sd=1)
        X = pymc3.Normal(name='X', mu=1, sd=2, observed=X)
        mu = alpha + beta.dot(X.T)
        Y_obs = pymc3.Normal(name='Y_obs', mu=mu, sd=sigma, observed=Y)

    return lr_model

TRUE_ALPHA, TRUE_SIGMA = 1, 1
TRUE_BETA = [1, 2.5]
size = 100
X1 = np.linspace(0, 1, size)
X2 = np.linspace(0,.2, size)
Y = TRUE_ALPHA + TRUE_BETA[0]*X1 + TRUE_BETA[1]*X2 + np.random.randn(size)*TRUE_SIGMA

X = np.array(list(zip(X1, X2)))
lr_model = build_lr_model(X, Y)

map_estimate = pymc3.find_MAP(model=lr_model)

print(map_estimate)



