import numpy as np
import linear_regression
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

def linear_regression_demo():
    TRUE_ALPHA, TRUE_SIGMA = 1, 1
    TRUE_BETA = 2.5
    size = 100
    X = np.linspace(0, 1, size)
    noise = (np.random.randn(size)*TRUE_SIGMA)
    y = (TRUE_ALPHA + TRUE_BETA*X + noise)

    lr = linear_regression.LinearRegression()
    lr.fit(X, y, 1000)
    plt.plot(X, y, linewidth=0.0, marker='x', color='g')
    pred_post_points = [lr.get_predictive_posterior_samples(x) for x in X]
    transpose = list(zip(*pred_post_points))
    for y_values in transpose:
        plt.plot(X, y_values, color='r')
    predicted_line = [lr.predict(x) for x in X]
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

    # Add outliers - Make this a different test, compare to
    # X = np.concatenate((X, [-.30]), axis=0)
    # y = np.concatenate((y, [5.0]), axis=0)
    # size += 1

    lr = linear_regression.LinearRegression()
    poly_X = PolynomialFeatures(include_bias=False, degree=4).fit_transform(X.reshape((size,1)))

    lr.fit(poly_X, y, 1000)
    pred_post_points = [lr.get_predictive_posterior_samples(x) for x in poly_X]
    transpose = list(zip(*pred_post_points))
    for y_values in transpose:
        plt.plot(X, y_values, color='r')
    predicted_line = [lr.predict(x) for x in poly_X]
    plt.plot(X, predicted_line)
    plt.plot(X, y, linewidth=0.0, marker='x', color='g')
    plt.show()

if __name__ == '__main__':
    # linear_regression_demo()
    polynomial_regression_demo()