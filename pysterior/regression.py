import functools
import itertools
import numpy as np
import pymc3
from pymc3 import stats

class _AbstractModel(object):
    def fit(self, X, y, sampling_iterations, sampler='NUTS'):
        X = self._force_shape(X)
        self.input_data_dimension = len(X[0])
        model = self._build_model(X, y)
        with model:
            self.map_estimate = pymc3.find_MAP(model=model)
            if sampler == 'NUTS':
                step = pymc3.NUTS(scaling=self.map_estimate)
            elif sampler == 'Metropolis':
                step = pymc3.Metropolis()
            else:
                raise Exception('Unrecognized sampler {0}'.format(sampler))
            trace = pymc3.sample(sampling_iterations, step, start=self.map_estimate)
        self.samples = trace

    def traceplot(self):
        pymc3.traceplot(self.get_samples(), vars=['alpha', 'beta'])

    def summary(self):
        pymc3.summary(self.get_samples(), vars=['alpha', 'beta'])

    def _build_model(self, X, y):
        raise NotImplementedError

    def _force_shape(self, X):
        shape = np.shape(X)
        if len(shape) == 1:
            return np.reshape(X, (shape[0], 1))
        return X

    def get_map_estimate(self):
        return self.map_estimate

    def get_samples(self):
        return self.samples

class _AbstractLinearRegression(_AbstractModel):
    def get_predictive_posterior_samples(self, x):
        "Obtain a sample of the output variable's distribution by running the sample variable values through the model."
        predictive_posterior_samples = []
        for alpha, beta in zip(self.samples['alpha'], self.samples['beta']):
            predictive_posterior_samples.append(alpha + np.dot(x, beta))
        return predictive_posterior_samples

    def predict_single(self, x):
        "Approximates the expected value of the output variable."
        s = self.get_predictive_posterior_samples(x)
        return sum(s) / len(s)

    def predict(self, X):
        return np.array([self.predict_single(x) for x in X])

    def predict_hpd_interval(self, X, alpha):
        return np.array([self.predict_hpd_interval_single(x, alpha) for x in X])

    def predict_hpd_interval_single(self, x, alpha):
        if alpha > 0.5:
            raise Exception('Invalid alpha: '.format(alpha))
        s = np.array(self.get_predictive_posterior_samples(x))
        return pymc3.stats.hpd(s, alpha)

    def predict_central_credible_interval(self, X, alpha):
        return np.array([self.predict_central_credible_interval_single(x, alpha) for x in X])

    def predict_central_credible_interval_single(self, x, alpha):
        if alpha > 0.5:
            raise Exception('Invalid alpha: '.format(alpha))
        s = self.get_predictive_posterior_samples(x)
        return self._get_central_credible_interval_from_sorted_samples(sorted(s), alpha)

    def _get_central_credible_interval_from_sorted_samples(self, samples, alpha):
        left_index = round((alpha/2.0) * len(samples))-1
        return (samples[left_index], samples[-left_index-1])

    def _build_model(self, X, y):
        lr_model = pymc3.Model()

        with lr_model:
            alpha = self._get_alpha()
            beta = self._get_beta()
            sigma = self._get_sigma()
            X = pymc3.Normal(name='X', mu=1, sd=2, observed=X)
            mu = alpha + beta.dot(X.T)
            Y_obs = pymc3.Normal(name='Y_obs', mu=mu, sd=sigma, observed=y)

        return lr_model

    def _get_sigma(self):
        noise_precision = pymc3.Uniform(name='noise_precision')
        sigma = pymc3.HalfNormal(name='sigma', sd=1.0 / noise_precision)
        return sigma

    def _get_alpha(self):
        raise NotImplementedError

    def _get_beta(self):
        raise NotImplementedError

class LinearRegression(_AbstractLinearRegression):
    def _get_alpha(self):
        alpha_precision = pymc3.Uniform(name='alpha_precision')
        alpha = pymc3.Normal(name='alpha', mu=0, sd=1.0 / alpha_precision)
        return alpha

    def _get_beta(self):
        precision = pymc3.Uniform(name='precision')
        beta = pymc3.Normal(name='beta', mu=0, sd=1.0 / precision, shape=self.input_data_dimension)
        return beta

class RidgeRegression(_AbstractLinearRegression):
    def __init__(self, weight_prior_sdev):
        self.weight_prior_sdev = weight_prior_sdev

    def _get_alpha(self):
        alpha = pymc3.Normal(name='alpha', mu=0, sd=self.weight_prior_sdev)
        return alpha

    def _get_beta(self):
        precision = pymc3.Uniform(name='precision')
        beta = pymc3.Normal(name='beta', mu=0, sd=1.0 / precision, shape=self.input_data_dimension)
        return beta

class LassoRegression(_AbstractLinearRegression):
    def __init__(self, weight_prior_scale):
        self.weight_prior_scale = weight_prior_scale

    def _get_alpha(self):
        alpha = pymc3.Laplace('alpha', 0, self.weight_prior_scale)
        return alpha

    def _get_beta(self):
        beta = pymc3.Laplace('beta', 0, self.weight_prior_scale, shape=self.input_data_dimension)
        return beta

class RobustLinearRegression(LinearRegression):
    def _get_sigma(self):
        noise_precision = pymc3.Uniform(name='noise_precision')
        sigma = pymc3.HalfCauchy(name='sigma', beta=1.0 / noise_precision)
        return sigma

class PolynomialRegression(LinearRegression):
    def __init__(self, degree, include_bias=True):
        self.degree = degree
        self.include_bias = True

    def _get_dimension_from_data_point(self, point):
        shape = np.shape(point)
        if not shape:
            return 1
        else:
            return shape[0]

    def fit(self, X, y, sampling_iterations):
        dimension = self._get_dimension_from_data_point(X[0])
        self.feature_generator = _PolynomialFeatureGenerator(self.degree, dimension, inclue_bias=self.include_bias)
        poly_X = np.array([self.feature_generator.preprocess(x) for x in X])
        return super(PolynomialRegression, self).fit(poly_X, y, sampling_iterations)

    def get_predictive_posterior_samples(self, x):
        poly_x = self.feature_generator.preprocess(x)
        return super(PolynomialRegression, self).get_predictive_posterior_samples(poly_x)

class _NondecreasingSequenceEnumerator(object):
    def get_ones_vector(self, sequence):
        if 0 in sequence:
            np_first_zero_index = list(np.where(sequence==0))
        else:
            np_first_zero_index = None
        if np_first_zero_index:
            first_zero = np_first_zero_index[0][0]
            ones_vector = np.array([1 if i < first_zero else 0
                                    for i in range(len(sequence))])
        else:
            min_value = sequence[-1]
            ones_vector = np.ones(len(sequence)) * min_value
        return ones_vector

    def get_inner_move(self, sequence):
        ones_vector = self.get_ones_vector(sequence)
        reduced_sequence = sequence - ones_vector
        reshaped_reduced_sequence = self.get_outer_move(reduced_sequence)
        moved_sequence = reshaped_reduced_sequence + ones_vector
        return moved_sequence

    def rindex(self, seq, target):
        rev = reversed(seq)
        for i, x in enumerate(rev):
            if x == target:
                return len(seq) - 1 - i
        return -1


    def get_outer_move(self, sequence):
        greatest = sequence[0]
        rightmost_greatest_index = self.rindex(sequence, greatest)
        first_zero = np.where(sequence==0)[0][0]
        sequence[rightmost_greatest_index] -= 1
        sequence[first_zero] += 1
        return sequence

    def is_valid(self, seq):
        for l,r in zip(seq[:-1], seq[1:]):
            if r > l:
                return False
        return True

    def inner_move_possible(self, seq):
        ones_vector = self.get_ones_vector(seq)
        reduced_vector = seq - ones_vector
        return self.outer_move_possible(reduced_vector)

    def outer_move_possible(self, sequence):
        has_zero = 0 not in sequence
        if has_zero or sequence[0] == 1 or sequence[0] == 0:
            return False
        return True

    def is_final_config(self, seq):
        return (not self.inner_move_possible(seq)) and (not self.outer_move_possible(seq))

    def non_increasing_sequences(self, l, n):
        initial_sequence = np.array([n] + [0] * (l-1))
        seq = initial_sequence
        sequences = [np.copy(seq)]
        while not self.is_final_config(seq):
            while self.inner_move_possible(seq):
                seq = self.get_inner_move(seq)
                sequences.append(np.copy(seq))
            if self.outer_move_possible(seq):
                seq = self.get_outer_move(seq)
                sequences.append(np.copy(seq))
        return sequences


class _PolynomialFeatureGenerator(object):
    def __init__(self, degree, dimension, inclue_bias=True):
        self.degree = degree
        self.include_bias = inclue_bias
        self._set_exponent_vectors(dimension)

    def _set_exponent_vectors(self, size):
        exponent_vectors = []
        if self.include_bias:
            exponent_vectors = [np.zeros(size)]
        for i in range(1,self.degree+1):
            base_exponents = _NondecreasingSequenceEnumerator().non_increasing_sequences(size, i)
            all_exponents_nested = [list(set(itertools.permutations(exponent))) for exponent in base_exponents]
            all_exponents = list(itertools.chain.from_iterable(all_exponents_nested))
            vectorized_exponents = [np.array(x) for x in all_exponents]
            exponent_vectors.extend(vectorized_exponents)
        self.exponent_vectors = exponent_vectors

    def _get_polynomial_term(self, x, exponents):
        if not np.shape(x):
            return x**exponents[0]
        product_terms = [base**ex for base,ex in zip(x, exponents)]
        product = lambda x1, x2: x1*x2
        return functools.reduce(product, product_terms)

    def preprocess(self, row):
        make_term_from_exponents = functools.partial(self._get_polynomial_term, row)
        poly_row = np.array(list(map(make_term_from_exponents, self.exponent_vectors)))
        return poly_row