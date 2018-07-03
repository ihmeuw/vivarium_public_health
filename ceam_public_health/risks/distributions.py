from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats, optimize, integrate, special


class NonConvergenceError(Exception):
    """ Raised when the optimization fails to converge """
    def __init__(self, message, dist):
        super().__init__(message)
        self.dist = dist


def _get_optimization_result(exposure, func, initial_func):
    """ It finds the shape parameters of distributions which generates mean/sd close to actual mean/sd
    Parameters
    ---------
    exposure : pd.DataFrame with 'mean' and 'standard_deviation'
    func: the objective function to minimize, arguments (initial guess, mean, sd)
    initial_func: lambda function to make a guess based on mean/sd

    Returns
    --------
    tuples of scipy.optimize.optimize.OptimizeResult
    """

    mean, sd = exposure['mean'].values, exposure['standard_deviation'].values
    results = tuple()
    with np.errstate(all='warn'):
        for i in range(len(mean)):
            initial_guess = initial_func(mean[i], sd[i])
            result = optimize.minimize(func, initial_guess, (mean[i], sd[i],), method='Nelder-Mead',
                                           options={'maxiter': 10000})
            results += (result,)
    return results


class BaseDistribution:
    def __init__(self, exposure):
        self._range = self._get_min_max(exposure)
        self._parameter_data = self._get_params(exposure)
        self._ranges_data = {'x_min': pd.DataFrame(self._range['x_min'], index=exposure.index),
                             'x_max': pd.DataFrame(self._range['x_max'], index=exposure.index)}

    def setup(self, builder):
        self.parameters = {name: builder.lookup.build_table(data.reset_index()) for name, data in
                           self._parameter_data.items()}
        self.ranges_data = {name: builder.lookup.build_table(data.reset_index()) for name, data in
                            self._ranges_data.items()}

    @staticmethod
    def _get_min_max(exposure: pd.DataFrame):
        exposure_mean, exposure_sd = exposure['mean'], exposure['standard_deviation']
        alpha = 1 + exposure_sd ** 2 / exposure_mean ** 2
        scale = exposure_mean / np.sqrt(alpha)
        s = np.sqrt(np.log(alpha))
        x_min = stats.lognorm(s=s, scale=scale).ppf(.001)
        x_max = stats.lognorm(s=s, scale=scale).ppf(.999)
        return {'x_min': x_min, 'x_max': x_max}

    def _get_params(self, exposure: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        # Should return dict of {param_name: param_dataframe}
        raise NotImplementedError()

    def process(self, data, process_type, ranges):
        return data

    def pdf(self, x, interpolation=True):
        if not interpolation:
            data_size = len(x)
            params = self._parameter_data
            ranges = {key: np.repeat(val, data_size) for key, val in self._range.items()}
        else:
            params = {name: p(x.index) for name, p in self.parameters.items()}
            ranges = {name: p(x.index) for name, p in self.ranges_data.items()}

        x = self.process(x, "pdf_preprocess", ranges)
        pdf = self.distribution(**params).pdf(x)
        return self.process(pdf, "pdf_postprocess", ranges)

    def ppf(self, x, interpolation=True):
        if not interpolation:
            data_size = len(x)
            group_size = len(self._range['x_min'])
            params = self._parameter_data
            ranges = {key: np.repeat(val, data_size).reshape(group_size, data_size) for key, val in self._range.items()}
        else:
            params = {name: p(x.index) for name, p in self.parameters.items()}
            ranges = {name: p(x.index) for name, p in self.ranges_data.items()}

        x = self.process(x, "ppf_preprocess", ranges)
        ppf = self.distribution(**params).ppf(x)
        return self.process(ppf, "ppf_postprocess", ranges)


class Beta(BaseDistribution):

    distribution = stats.beta

    def _get_params(self, exposure):
        exposure_mean, exposure_sd = exposure['mean'], exposure['standard_deviation']
        scale = self._range['x_max'] - self._range['x_min']
        a = 1 / scale * (exposure_mean - self._range['x_min'])
        b = (1 / scale * exposure_sd) ** 2
        shape_1 = a ** 2 / b * (1 - a) - a
        shape_2 = a / b * (1 - a) ** 2 + (a - 1)
        params = {'scale': pd.DataFrame(scale, index=exposure.index),
                  'a': pd.DataFrame(shape_1, index=exposure.index),
                  'b': pd.DataFrame(shape_2, index=exposure.index)}
        return params

    def process(self, data, process_type, ranges):
        if process_type == 'pdf_preprocess':
            return data - ranges['x_min']
        elif process_type == 'ppf_postprocess':
            return data + ranges['x_max'] - ranges['x_min']
        else:
            return data


class Exponential(BaseDistribution):

    distribution = stats.expon

    def _get_params(self, exposure):
        return {'scale': pd.DataFrame(exposure['mean'], index=exposure.index)}


class Gamma(BaseDistribution):

    distribution = stats.gamma

    def _get_params(self, exposure):
        mean, sd = exposure['mean'], exposure['standard_deviation']
        a = (mean / sd) ** 2
        scale = sd ** 2 / mean
        return {'a': pd.DataFrame(a, index=exposure.index), 'scale': pd.DataFrame(scale, index=exposure.index)}


class GeneralizedLogNormal(BaseDistribution):
    """Here for completeness.

    The weight for the glnorm distribution is zero for all ensemble modeled risks for 2016.  Which is good, because
    getting the parameters involves running some optimization technique, and I definitely don't want to spend
    time ensuring R's optimization scheme is equivalent to scipy's.  - J.C.
    """
    def pdf(self, _):
        return 0

    def ppf(self, _):
        return 0


class Gumbel(BaseDistribution):

    distribution = stats.gumbel_r

    def _get_params(self, exposure):
        mean, sd = exposure['mean'], exposure['standard_deviation']
        loc = mean - (np.euler_gamma * np.sqrt(6) / np.pi * sd)
        scale = np.sqrt(6) / np.pi * sd
        return {'loc': pd.DataFrame(loc, index=exposure.index),
                'scale': pd.DataFrame(scale, index=exposure.index)}


class InverseGamma(BaseDistribution):

    distribution = stats.invgamma

    def _get_params(self, exposure):
        def f(guess, mean, sd):
            alpha, beta = np.abs(guess)
            mean_guess = beta / (alpha - 1)
            var_guess = beta ** 2 / ((alpha - 1) ** 2 * (alpha - 2))
            return (mean - mean_guess) ** 2 + (sd ** 2 - var_guess) ** 2

        opt_results = _get_optimization_result(exposure, f, lambda m, s: np.array((m, m * s)))
        data_size = len(exposure)
        try:
            assert np.all([opt_results[k].success is True for k in range(data_size)])
            shape = np.abs([opt_results[k].x[0] for k in range(data_size)])
            scale = np.abs([opt_results[k].x[1] for k in range(data_size)])
            return {'a': pd.DataFrame(shape, index=exposure.index), 'scale': pd.DataFrame(scale, index=exposure.index)}

        except AssertionError:
            raise NonConvergenceError('InverseGamma did not converge!!', 'invgamma')


class InverseWeibull(BaseDistribution):

    distribution = stats.invweibull

    def _get_params(self, exposure):
        # moments from  Stat Papers (2011) 52: 591. https://doi.org/10.1007/s00362-009-0271-3
        # it is much faster than using stats.invweibull.mean/var
        def f(guess, mean, sd):
            shape, scale = np.abs(guess)
            mean_guess = scale * special.gamma(1 - 1 / shape)
            var_guess = scale ** 2 * special.gamma(1 - 2 / shape) - mean_guess ** 2
            return (mean - mean_guess) ** 2 + (sd ** 2 - var_guess) ** 2

        opt_results = _get_optimization_result(exposure, f, lambda m, s: np.array((max(2.2, s / m), m)))
        data_size = len(exposure)
        try:
            assert np.all([opt_results[k].success is True for k in range(data_size)])
            shape = np.abs([opt_results[k].x[0] for k in range(data_size)])
            scale = np.abs([opt_results[k].x[1] for k in range(data_size)])
            return {'c': pd.DataFrame(shape, index=exposure.index), 'scale': pd.DataFrame(scale, index=exposure.index)}
        except AssertionError:
            raise NonConvergenceError('InverseWeibull did not converge!!', 'invweibull')


class LogLogistic(BaseDistribution):

    distribution = stats.burr12

    def _get_params(self, exposure):
        def f(guess, mean, sd):
            shape, scale = np.abs(guess)
            b = np.pi / shape
            mean_guess = scale * b / np.sin(b)
            var_guess = scale ** 2 * 2 * b / np.sin(2 * b) - mean_guess ** 2
            return (mean - mean_guess) ** 2 + (sd ** 2 - var_guess) ** 2

        opt_results = _get_optimization_result(exposure, f, lambda m, s: np.array((max(2, m), m)))
        data_size =len(exposure)
        try:
            assert np.all([opt_results[k].success is True for k in range(data_size)])
            shape = np.abs([opt_results[k].x[0] for k in range(data_size)])
            scale = np.abs([opt_results[k].x[1] for k in range(data_size)])
            return {'c': pd.DataFrame(shape, index=exposure.index),
                    'd': pd.DataFrame([1]*len(exposure), index=exposure.index),
                    'scale': pd.DataFrame(scale, index=exposure.index)}
        except AssertionError:
            raise NonConvergenceError('LogLogistic did not converge!!', 'llogis')


class LogNormal(BaseDistribution):

    distribution = stats.lognorm

    def _get_params(self, exposure):
        mean, sd = exposure['mean'], exposure['standard_deviation']
        alpha = 1 + sd ** 2 / mean ** 2
        s = np.sqrt(np.log(alpha))
        scale = mean / np.sqrt(alpha)
        return {'s': pd.DataFrame(s, index=exposure.index),
                'scale': pd.DataFrame(scale, index=exposure.index)}


class MirroredGumbel(BaseDistribution):

    distribution = stats.gumbel_r

    def _get_params(self, exposure):
        loc = self._range['x_max'] - exposure['mean'] - (
                    np.euler_gamma * np.sqrt(6) / np.pi * exposure['standard_deviation'])
        scale = np.sqrt(6) / np.pi * exposure['standard_deviation']
        return {'loc': pd.DataFrame(loc, index=exposure.index),
                'scale': pd.DataFrame(scale, index=exposure.index)}

    def process(self, data, process_type, ranges):
        if process_type == 'pdf_preprocess':
            return ranges['x_max'] - data
        elif process_type == 'ppf_preprocess':
            return np.tile(1, data.shape) - data
        elif process_type == 'ppf_postprocess':
            return ranges['x_max'] - data
        else:
            return data


class MirroredGamma(BaseDistribution):

    distribution = stats.gamma

    def _get_params(self, exposure):
        mean, sd = exposure['mean'], exposure['standard_deviation']
        a = ((self._range['x_max'] - mean) / sd) ** 2
        scale = sd ** 2 / (self._range['x_max'] - mean)
        return {'a': pd.DataFrame(a, index=exposure.index), 'scale': pd.DataFrame(scale, index=exposure.index)}

    def process(self, data, process_type, ranges):
        if process_type == 'pdf_preprocess':
            return ranges['x_max'] - data
        elif process_type == 'ppf_preprocess':
            return np.tile(1, data.shape) - data
        elif process_type == 'ppf_postprocess':
            return ranges['x_max'] - data
        else:
            return data


class Normal(BaseDistribution):

    distribution = stats.norm

    def _get_params(self, exposure):
        return {'loc': pd.DataFrame(exposure['mean'], index=exposure.index),
                'scale': pd.DataFrame(exposure['standard_deviation'], index=exposure.index)}


class Weibull(BaseDistribution):

    distribution = stats.weibull_min

    def _get_params(self, exposure):
        def f(guess, mean, sd):
            shape, scale = np.abs(guess)
            mean_guess = scale * special.gamma(1 + 1 / shape)
            var_guess = scale ** 2 * special.gamma(1 + 2 / shape) - mean_guess ** 2
            return (mean - mean_guess) ** 2 + (sd ** 2 - var_guess) ** 2

        opt_results = _get_optimization_result(exposure, f, lambda m, s: np.array((m, m / s)))
        data_size = len(exposure)
        try:
            assert np.all([opt_results[k].success is True for k in range(data_size)])
            shape = np.abs([opt_results[k].x[0] for k in range(data_size)])
            scale = np.abs([opt_results[k].x[1] for k in range(data_size)])
            return {'c': pd.DataFrame(shape, index=exposure.index), 'scale': pd.DataFrame(scale, index=exposure.index)}

        except AssertionError:
            raise NonConvergenceError('Weibull did not converge!!', 'weibull')


class EnsembleDistribution:

    def __init__(self, exposure, weights, distribution_map):
        self.weights, self._distributions = self.get_valid_distributions(exposure, weights, distribution_map)

    @staticmethod
    def get_valid_distributions(exposure, weights, maps):
        """
        :param exposure: pd.DataFrame with columns=['mean', 'standard_deviation']
        :param weights: pd.Series with distribution key and weights
        :param maps: dictionary form of distribution key and class

        :return
        weights: rescaled weigths after dropping non-convergence distribution (pd.Series)
        dist: subset of maps with converged distributions only
        """
        dist = dict()
        for key in maps:
            try:
                dist[key] = maps[key](exposure)
            except NonConvergenceError as e:
                if weights[e.dist] > 0.05:
                    weights = weights.drop(e.dist)
                else:
                    # if the weight is larger than 5%, we can't drop the distribution. Re-raise the error.
                    raise NonConvergenceError(f'Divergent {key} distribution has weights: {100*weights[key]}%', key)
                pass

        return weights/np.sum(weights), dist

    def setup(self, builder):
        builder.components.add_components([self._distributions[dist_name] for dist_name in self._distributions.keys()])

    def pdf(self, x, interpolation=True):
        return np.sum([self.weights[dist_name] * self._distributions[dist_name].pdf(x, interpolation) for dist_name in
                       self._distributions.keys()], axis=0)

    def ppf(self, x, interpolation=True):
        return np.sum([self.weights[dist_name] * self._distributions[dist_name].ppf(x, interpolation) for dist_name in
                       self._distributions.keys()], axis=0)


def get_distribution(risk, risk_type, builder):

    distribution = builder.data.load(f"{risk_type}.{risk}.distribution")
    exposure_mean = builder.data.load(f"{risk_type}.{risk}.exposure")
    exposure_sd = builder.data.load(f"{risk_type}.{risk}.exposure_standard_deviation")
    exposure_mean = exposure_mean.rename(index=str, columns={"value": "mean"})
    exposure_sd = exposure_sd.rename(index=str, columns={"value": "standard_deviation"})

    exposure = exposure_mean.merge(exposure_sd).set_index(['age', 'sex', 'year'])

    if distribution == 'ensemble':
        weights = builder.data.load(f'risk_factor.{risk}.ensemble_weights')
        distribution_map = {'betasr': Beta,
                            'exp': Exponential,
                            'gamma': Gamma,
                            'gumbel': Gumbel,
                            'invgamma': InverseGamma,
                            'invweibull': InverseWeibull,
                            'llogis': LogLogistic,
                            'lnorm': LogNormal,
                            'mgamma': MirroredGamma,
                            'mgumbel': MirroredGumbel,
                            'norm': Normal,
                            'weibull': Weibull}

        if risk == 'high_ldl_cholesterol':
            weights = weights.drop('invgamma', axis=1)

        # we drop the invweibull if its weight is less than 5 percent
        if 'invweibull' in weights.columns and np.all(weights['invweibull']< 0.05):
            weights = weights.drop('invweibull', axis=1)

        weights_cols = list(set(distribution_map.keys()) & set(weights.columns))
        weights = weights[weights_cols]

        # weight is all same across the demo groups
        e_weights = weights.iloc[0]

        dist = {d: distribution_map[d] for d in weights_cols}
        return EnsembleDistribution(exposure, e_weights/np.sum(e_weights), dist)

    elif distribution == 'lognormal':
        return LogNormal(exposure, stats.lognorm)
    elif distribution == 'normal':
        return Normal(exposure, stats.norm)
    else:
        raise ValueError(f"Unhandled distribution type {distribution}")
