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

def test_logistic_regression():
    data_X = np.array([[0,0],[1,0],[0,1],[1,1]])
    data_y = np.array([0,1,0,1])

    with pymc3.Model() as model:
        X = pymc3.Normal(name='X', mu=1, sd=2, observed=data_X)
        alpha1_precision = pymc3.Uniform(name='alpha1_precision')
        alpha1 = pymc3.Normal(name='alpha1', mu=0, sd=1.0 / alpha1_precision)
        alpha2_precision = pymc3.Uniform(name='alpha2_precision')
        alpha2 = pymc3.Normal(name='alpha2', mu=0, sd=1.0 / alpha2_precision)
        beta1_precision = pymc3.Uniform(name='beta1_precision')
        beta2_precision = pymc3.Uniform(name='beta2_precision')
        beta1 = pymc3.Normal(name='beta1', mu=0.0, sd=1.0 / beta1_precision, shape=2)
        beta2 = pymc3.Normal(name='beta2', mu=0.0, sd=1.0 / beta2_precision, shape=2)
        v1 = alpha1 + beta1.dot(X.T)
        v2 = alpha2 + beta2.dot(X.T)
        denom = T.exp(v1) + T.exp(v2)
        sm1 = T.exp(v1) / denom
        sm2 = T.exp(v2) / denom
        v = T.stack(sm1, sm2)
        y = pymc3.Categorical(p=v, name='y', observed=data_y, shape=len(data_y))

        step = pymc3.Metropolis()
        trace = pymc3.sample(1000, step)

    pass

def test_mlp():
    true_beta = np.array([-3.0, 5.0])
    true_alpha = 5.0
    data_X = np.array(list(zip(np.linspace(-1.0, 1.0),np.linspace(-1.0, 1.0))))
    data_y = np.array([true_beta.dot(x) + true_alpha for x in data_X])
    n_inputs = 2
    n_hidden = 5

    with pymc3.Model() as model:
        X = pymc3.Normal(name='X', mu=1, sd=2, observed=data_X)
        w1_precision = pymc3.Uniform(name='w1_precision')
        w1 = pymc3.Normal(name='w1', mu=0, sd=1.0 / w1_precision, shape=(n_hidden, n_inputs))
        alpha_precision = pymc3.Uniform(name='alpha_precision')
        alpha = pymc3.Normal(name='alpha', mu=0, sd=1.0 / alpha_precision, shape=n_hidden)
        h = T.nnet.sigmoid(w1.dot(X.T) + alpha)
        w2_precision = pymc3.Uniform(name='w2_precision')
        w2 = pymc3.Normal(name='w2', mu=0, sd = 1.0 / w2_precision, shape=n_inputs)
        output_noise_precision = pymc3.Uniform(name='output_noise_precision')
        y = pymc3.Normal(name='y', mu=h.dot(w2.T), sd=1.0 / output_noise_precision, observed=data_y)

        step = pymc3.Metropolis()
        trace = pymc3.sample(1000, step)


if __name__ == '__main__':
    # test_bernoulli()
    # test_categorical()
    # test_poisson_analysis()
    # test_logistic_regression()
    test_mlp()