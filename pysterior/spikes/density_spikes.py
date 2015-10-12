import pymc3
import random
from matplotlib import pyplot as plt
import numpy as np
import theano.tensor as T

def test_bernoulli():
    data = [random.randint(0,1) for i in range(200)]

    model = pymc3.Model()

    with model:
        p = pymc3.Uniform(lower=0,upper=1, name='p')
        X = pymc3.Bernoulli(p=p, name='X', observed=data)

        start = pymc3.find_MAP()

        # instantiate sampler
        step = pymc3.NUTS(scaling=start)

        # draw 500 posterior samples
        trace = pymc3.sample(10000, step, start=start)

    pymc3.traceplot(trace)
    plt.show()

def test_categorical():
    k = 3
    ndata = 5000

    v = np.random.randint(0, k, ndata)

    with pymc3.Model() as model:
        p = pymc3.Dirichlet(name='p', a=np.array([1., 1., 1.]), shape=k)
        category = pymc3.Categorical(name='category',
                                     p=p,
                                     shape=ndata,
                                     observed=v)
        step = pymc3.Metropolis(vars=[p, category])
        trace = pymc3.sample(3000, step=step)

    pymc3.traceplot(trace)
    plt.show()

def test_poisson_analysis():
    # Synthetic data
    # data_1 = np.random.poisson(lam=2.0, size=100)
    # data_2 = np.random.poisson(lam=5.0, size=100)
    all_data = [ 4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6, # Coal mining data
                   3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
                   2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0, 0,
                   1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
                   0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
                   3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
                   0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1]
    i=43 # Is the rate after the switchpoint different?
    data_1, data_2 = all_data[:i], all_data[i:]
    with pymc3.Model() as model:
        mu1 = pymc3.Uniform(name='mu1', lower=0,upper=1)
        mu2 = pymc3.Uniform(name='mu2', lower=0,upper=1)
        mu1_corrected = mu1**-1
        mu2_corrected = mu2**-1
        X1 = pymc3.Poisson(name='X1', mu=mu1_corrected, observed=data_1)
        X2 = pymc3.Poisson(name='X2', mu=mu2_corrected, observed=data_2)

        step = pymc3.NUTS()
        trace = pymc3.sample(10000, step=step, )

    # pymc3.traceplot(trace)
    mu1_adjusted = trace['mu1']**-1
    mu2_adjusted = trace['mu2']**-1
    plt.hist(mu1_adjusted - mu2_adjusted, bins=50)
    # Difference distribution
    # P-value?
    # Type I and Type II error likelihoods
    plt.show()

if __name__ == '__main__':
    # test_bernoulli()
    # test_categorical()
    test_poisson_analysis()