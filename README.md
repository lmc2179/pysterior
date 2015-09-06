# pysterior

`pysterior` is a machine learning library for Python which aims to make Bayesian parametric regression and classification models accessible and easy to use. The library allows users to construct supervised learning models using an intuitive interface similar to that used by [scikit-learn](https://github.com/scikit-learn/scikit-learn).

Under the hood, `pysterior` uses the implementation of the [No-U-Turn Sampler (Hoffman and Gelman, 2011)](http://arxiv.org/abs/1111.4246) provided by the thoroughly wonderful [pymc3](https://github.com/pymc-devs/pymc3). Pymc3 is a requirement to run `pysterior`; see the pymc3 page to find out how to get the latest version.

You can install the latest version of this package from PyPI, where it is currently in alpha. The current models are currently supported:
* Bayesian Linear Rigression
* Bayesian Ridge Regression
* Bayesian Lasso Regression
* Bayesian Robust Linear regression (Cauchy-distributed noise)

Coming soon:
* Logistic regression
* Neural networks
* Examples of regression models with `pysterior`  
