import numpy as np

from pysterior import regression
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

def linear_regression_demo():
    TRUE_ALPHA, TRUE_SIGMA = 1, 1
    TRUE_BETA = 2.5
    size = 100
    X = np.linspace(0, 1, size)
    noise = (np.random.randn(size)*TRUE_SIGMA)
    y = (TRUE_ALPHA + TRUE_BETA*X + noise)

    lr = regression.LinearRegression()
    lr.fit(X, y, 1000)
    plt.plot(X, y, linewidth=0.0, marker='x', color='g')
    pred_post_points = [lr.get_predictive_posterior_samples(x) for x in X]
    transpose = list(zip(*pred_post_points))
    for y_values in transpose:
        plt.plot(X, y_values, color='r')
    predicted_line = lr.predict(X)
    plt.plot(X, predicted_line)
    plt.show()

def linear_regression_credible_demo():
    TRUE_ALPHA, TRUE_SIGMA = 1, 1
    TRUE_BETA = 2.5
    size = 100
    X = np.linspace(0, 1, size)
    noise = (np.random.randn(size)*TRUE_SIGMA)
    y = (TRUE_ALPHA + TRUE_BETA*X + noise)

    lr = regression.LinearRegression()
    lr.fit(X, y, 2000)
    plt.plot(X, y, linewidth=0.0, marker='x', color='g')
    for alpha, color in [(0.05, 'r'), (0.1, 'g'), (0.2, 'b')]:
        intervals = lr.predict_central_credible_interval(X, alpha)
        lower_bound, upper_bound = zip(*intervals)
        plt.plot(X, lower_bound, color=color)
        plt.plot(X, upper_bound, color=color)
    predicted_line = lr.predict(X)
    plt.plot(X, predicted_line)
    plt.show()

def polynomial_regression_demo():
    TRUE_ALPHA, TRUE_SIGMA = 1, 1.0
    TRUE_BETA1 = 2.5
    TRUE_BETA2 = 6.5
    TRUE_BETA3 = 2.5
    TRUE_BETA4 = -10.5
    size = 10
    X = np.linspace(-1.0, 1.0, size)
    noise = (np.random.randn(size)*TRUE_SIGMA)
    y = (TRUE_ALPHA + TRUE_BETA1*X + TRUE_BETA2*X**2 + TRUE_BETA3*X**3  + TRUE_BETA4*X**4 + noise)

    lr = regression.PolynomialRegression(4)
    lr.fit(X, y, 1000)
    pred_post_points = [lr.get_predictive_posterior_samples(x) for x in X]
    transpose = list(zip(*pred_post_points))
    for y_values in transpose:
        plt.plot(X, y_values, color='r')
    predicted_line = lr.predict(X)
    plt.plot(X, predicted_line)
    plt.plot(X, y, linewidth=0.0, marker='x', color='g')
    plt.show()

def polynomial_regression_interval_demo():
    TRUE_ALPHA, TRUE_SIGMA = 1, 1.0
    TRUE_BETA1 = 2.5
    TRUE_BETA2 = 6.5
    TRUE_BETA3 = 2.5
    TRUE_BETA4 = -10.5
    size = 10
    X = np.linspace(-1.0, 1.0, size)
    noise = (np.random.randn(size)*TRUE_SIGMA)
    y = (TRUE_ALPHA + TRUE_BETA1*X + TRUE_BETA2*X**2 + TRUE_BETA3*X**3  + TRUE_BETA4*X**4 + noise)

    holdout_X = np.array([0.23, 0.78])
    noise = (np.random.randn(len(holdout_X))*TRUE_SIGMA)
    holdout_y = (TRUE_ALPHA + TRUE_BETA1*holdout_X + TRUE_BETA2*holdout_X**2 + TRUE_BETA3*holdout_X**3  + TRUE_BETA4*holdout_X**4 + noise)

    lr = regression.PolynomialRegression(degree=4)
    lr.fit(X, y, 7000)

    for alpha, color in [(0.05, 'r'), (0.1, 'g'), (0.2, 'b')]:
        intervals = lr.predict_central_credible_interval(X, alpha)
        lower_bound, upper_bound = zip(*intervals)
        plt.plot(X, lower_bound, color=color)
        plt.plot(X, upper_bound, color=color)

    predicted_line = lr.predict(X)
    plt.plot(X, predicted_line)
    plt.plot(X, y, linewidth=0.0, marker='x', color='g')
    plt.plot(holdout_X, holdout_y, linewidth=0.0, marker='x', color='r')
    plt.show()

def polynomial_regression_hpd_interval_demo():
    TRUE_ALPHA, TRUE_SIGMA = 1, 1.0
    TRUE_BETA1 = 2.5
    TRUE_BETA2 = 6.5
    TRUE_BETA3 = 2.5
    TRUE_BETA4 = -10.5
    size = 10
    X = np.linspace(-1.0, 1.0, size)
    noise = (np.random.randn(size)*TRUE_SIGMA)
    y = (TRUE_ALPHA + TRUE_BETA1*X + TRUE_BETA2*X**2 + TRUE_BETA3*X**3  + TRUE_BETA4*X**4 + noise)

    holdout_X = np.array([0.23, 0.78])
    noise = (np.random.randn(len(holdout_X))*TRUE_SIGMA)
    holdout_y = (TRUE_ALPHA + TRUE_BETA1*holdout_X + TRUE_BETA2*holdout_X**2 + TRUE_BETA3*holdout_X**3  + TRUE_BETA4*holdout_X**4 + noise)

    lr = regression.PolynomialRegression(degree=4)
    lr.fit(X, y, 7000)

    for alpha, color in [(0.05, 'r'), (0.1, 'g'), (0.2, 'b')]:
        intervals = lr.predict_hpd_interval(X, alpha)
        lower_bound, upper_bound = zip(*intervals)
        plt.plot(X, lower_bound, color=color)
        plt.plot(X, upper_bound, color=color)

    predicted_line = lr.predict(X)
    plt.plot(X, predicted_line)
    plt.plot(X, y, linewidth=0.0, marker='x', color='g')
    plt.plot(holdout_X, holdout_y, linewidth=0.0, marker='x', color='r')
    plt.show()

def cubic_regression_comparison():
    TRUE_ALPHA, TRUE_SIGMA = 1, 1.0
    TRUE_BETA1 = 2.5
    TRUE_BETA2 = 6.5
    TRUE_BETA3 = 2.5
    TRUE_BETA4 = -10.5
    size = 10
    X = np.linspace(-1.0, 1.0, size)
    noise = (np.random.randn(size)*TRUE_SIGMA)
    y = (TRUE_ALPHA + TRUE_BETA1*X + TRUE_BETA2*X**2 + TRUE_BETA3*X**3  + TRUE_BETA4*X**4 + noise)

    lr = regression.LinearRegression()
    poly_X = PolynomialFeatures(include_bias=False, degree=4).fit_transform(X.reshape((size,1)))

    lr.fit(poly_X, y, 1000)
    pred_post_points = [lr.get_predictive_posterior_samples(x) for x in poly_X]
    transpose = list(zip(*pred_post_points))
    for y_values in transpose:
        plt.plot(X, y_values, color='r', linestyle='dotted')
    predicted_line = [lr.predict(x) for x in poly_X]
    plt.plot(X, predicted_line)
    plt.plot(X, y, linewidth=0.0, marker='x', color='g')

    ols = linear_model.LinearRegression()
    ols.fit(poly_X, y)
    predicted_line = [ols.predict(x) for x in poly_X]
    plt.plot(X, predicted_line, color='c')

    plt.plot(X, y, linewidth=0.0, marker='x', color='g')
    plt.show()

def robust_cubic_regression_comparison():
    TRUE_ALPHA, TRUE_SIGMA = 1, 1.0
    TRUE_BETA1 = 2.5
    TRUE_BETA2 = 6.5
    TRUE_BETA3 = 2.5
    TRUE_BETA4 = -10.5
    size = 12
    X = np.linspace(-1.0, 1.0, size)
    noise = (np.random.randn(size)*TRUE_SIGMA)
    y = TRUE_ALPHA + TRUE_BETA1*X + TRUE_BETA2*X**2 + TRUE_BETA3*X**3  + TRUE_BETA4*X**4 + noise
    y[3] = -10.0
    y[4] = -8.0

    lr = regression.RobustLinearRegression()
    poly_X = PolynomialFeatures(include_bias=False, degree=4).fit_transform(X.reshape((size,1)))

    lr.fit(poly_X, y, 1000)
    pred_post_points = [lr.get_predictive_posterior_samples(x) for x in poly_X]
    transpose = list(zip(*pred_post_points))
    for y_values in transpose:
        plt.plot(X, y_values, color='r', linestyle='dotted')
    predicted_line = np.array([lr.predict(x) for x in poly_X])
    plt.plot(X, predicted_line, color='black')

    ols = linear_model.LinearRegression()
    ols.fit(poly_X, y)
    predicted_line = ols.predict(poly_X)
    plt.plot(X, predicted_line, color='blue')

    plt.plot(X, y, linewidth=0.0, marker='x', color='g')
    plt.show()

if __name__ == '__main__':
    # linear_regression_demo()
    # linear_regression_credible_demo()
    # polynomial_regression_interval_demo()
    polynomial_regression_demo()
    # cubic_regression_comparison()
    # robust_cubic_regression_comparison()
    # polynomial_regression_hpd_interval_demo()