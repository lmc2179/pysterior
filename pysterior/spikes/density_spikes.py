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

if __name__ == '__main__':
    # test_bernoulli()
    test_categorical()