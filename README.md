# pysterior

`pysterior` is a machine learning library for Python which aims to make Bayesian parametric regression and classification models accessible and easy to use. The library allows users to construct supervised learning models using an intuitive interface similar to that used by [scikit-learn](https://github.com/scikit-learn/scikit-learn).

**More documentation is to come - this is still a very new project.**

Under the hood, `pysterior` uses the implementation of the [No-U-Turn Sampler (Hoffman and Gelman, 2011)](http://arxiv.org/abs/1111.4246) provided by the thoroughly wonderful [pymc3](https://github.com/pymc-devs/pymc3). Pymc3 is a requirement to run `pysterior`; see the pymc3 page to find out how to get the latest version.

You can install the latest version of this package from PyPI, where it is currently in alpha. The following regression models are currently supported:
* (Bayesian) Linear Regression
* Ridge Regression
* Lasso Regression
* Robust Linear regression (Cauchy-distributed noise)
* Polynomial regression
* Binary Logistic regression

Coming soon:
* k-class Logistic regression
* More non-linear models

---------------------------------

For demos, checkout the [demos directory](https://github.com/lmc2179/pysterior/tree/master/pysterior/demo).

Simple linear regression example:

![Linear Regression](https://raw.githubusercontent.com/lmc2179/pysterior/master/pysterior/demo/simple_linear_regression.png)


```
    TRUE_ALPHA, TRUE_SIGMA = 1, 1
    TRUE_BETA = 2.5
    size = 100
    X = np.linspace(0, 1, size)
    noise = (np.random.randn(size)*TRUE_SIGMA)
    y = (TRUE_ALPHA + TRUE_BETA*X + noise)
    
    lr = regression.LinearRegression()
    lr.fit(X, y, 5000)
    plt.plot(X, y, linewidth=0.0, marker='x', color='g')
    pred_post_points = [lr.get_predictive_posterior_samples(x) for x in X]
    transpose = list(zip(*pred_post_points))
    for y_values in transpose:
        plt.plot(X, y_values, color='r')
    predicted_line = lr.predict(X)
    plt.plot(X, predicted_line)
    plt.show()
```

---------------------------------

Versions released:
* 0.1.0 - first release, Linear regression models
* 0.1.1 - predict() takes 2-dimensional array instead of single data point; added predict_central_credible_interval(X, alpha) for predicting central credible intervales of output
* 0.1.2 - added Polynomial regression
* 0.1.3 - added Highest Posterior Density Interval (HPDI) estimation for regression models
* 0.1.4 - added access to pymc3 traceplot and summary from model object; allowed manual selection of metropolis sampler if desired
