import numpy as np
import linear_regression
from matplotlib import pyplot as plt

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

if __name__ == '__main__':
    linear_regression_demo()