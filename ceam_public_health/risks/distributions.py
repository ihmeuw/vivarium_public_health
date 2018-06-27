import numpy as np
import pandas as pd
from scipy import stats, optimize, integrate, special


def _get_shape_scale(exposure_mean, exposure_sd, func, initial_func):
    """ initial guess from initial func = guess for (shape, scale)
        output= (DataFrame(shape), DataFrame(scale))
    """
    df_shape = exposure_mean.shape  # exposure_sd should have the same shape
    df_index = exposure_mean.index

    shape_cols = ['shape']
    scale_cols = ['scale']

    mean_matrix, sd_matrix = exposure_mean.values, exposure_sd.values
    mean, sd = np.ndarray.flatten(mean_matrix), np.ndarray.flatten(sd_matrix)
    shape, scale = np.empty(len(mean)), np.empty(len(sd))
    with np.errstate(all='warn'):
        for i in range(len(mean)):
            initial_guess = initial_func(mean[i], sd[i])
            try:
                result = optimize.minimize(func, initial_guess, (mean[i], sd[i],), method='Nelder-Mead',
                                           options={'maxiter': 10000})
                assert result.success
                shape[i], scale[i] = result.x[0], result.x[1]
            except AssertionError:
                print('DID NOT CONVERGE')
                return None

    return {'shape': pd.DataFrame(shape.reshape(df_shape), columns=shape_cols, index=df_index),
            'scale': pd.DataFrame(scale.reshape(df_shape), columns=scale_cols, index=df_index)}


class BaseDistribution:
    def __init__(self, exposure, dist):
        self._range = self._get_min_max(exposure)
        self._parameter_data = self._get_params(exposure)
        self.distribution = dist
        self._ranges_data = {'x_min': pd.DataFrame(self._range['x_min'], index=exposure.index),
                            'x_max': pd.DataFrame(self._range['x_max'], index=exposure.index)}

    def setup(self, builder):
        self.parameters = {name: builder.lookup.build_table(data.reset_index()) for name, data in
                           self._parameter_data.items()}
        self.ranges_data = {name: builder.lookup.build_table(data.reset_index()) for name, data in
                            self._ranges_data.items()}

    @staticmethod
    def _get_min_max(exposure):
        exposure_mean, exposure_sd = exposure['mean'], exposure['standard_deviation']
        alpha = 1 + exposure_sd ** 2 / exposure_mean ** 2
        scale = exposure_mean / np.sqrt(alpha)
        s = np.sqrt(np.log(alpha))
        x_min = stats.lognorm(s=s, scale=scale).ppf(.001)
        x_max = stats.lognorm(s=s, scale=scale).ppf(.999)
        return {'x_min': x_min, 'x_max': x_max}

    def _get_params(self, exposure):
        # Should return dict of {param_name: param_dataframe}
        raise NotImplementedError()

    def process(self, data, process_type, ranges):
        return data

    def pdf(self, x):
        params = {name: p(x.index) for name, p in self.parameters.items()}
        ranges = {name: p(x.index) for name, p in self.ranges_data.items()}
        x = self.process(x, "pdf_preprocess", ranges)
        pdf = self.distribution(**params).pdf(x)
        return self.process(pdf, "pdf_postprocess", ranges)

    def ppf(self, x):
        params = {name: p(x.index) for name, p in self.parameters.items()}
        ranges = {name: p(x.index) for name, p in self.ranges_data.items()}
        x = self.process(x, "ppf_preprocess", ranges)
        ppf = self.distribution(**params).ppf(x)
        return self.process(ppf, "ppf_postprocess", ranges)

    def pdf_test(self, x):
        params = self._parameter_data
        ranges = self._ranges_data
        x = self.process(x, "pdf_preprocess", ranges)
        pdf = self.distribution(**params).pdf(x)
        return self.process(pdf, "pdf_postprocess", ranges)

    def ppf_test(self, x):
        params = self._parameter_data
        ranges = self._ranges_data
        x = self.process(x, "ppf_preprocess", ranges)
        ppf = self.distribution(**params).ppf(x)
        return self.process(ppf, "ppf_postprocess", ranges)


class Beta(BaseDistribution):
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
    def _get_params(self, exposure):
        return {'scale': pd.DataFrame(exposure['mean'], index=exposure.index)}


class Gamma(BaseDistribution):
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
    """ Gumbel in R is gumbel_r in scipy """
    def _get_params(self, exposure):
        mean, sd = exposure['mean'], exposure['standard_deviation']
        loc = mean - (np.euler_gamma * np.sqrt(6) / np.pi * sd)
        scale = np.sqrt(6) / np.pi * sd
        return {'loc': pd.DataFrame(loc, index=exposure.index),
                'scale': pd.DataFrame(scale, index=exposure.index)}


class InverseGamma(BaseDistribution):
    def _get_params(self, exposure):
        def f(guess, mean, sd):
            alpha, beta = np.abs(guess)
            mean_guess = beta / (alpha - 1)
            var_guess = beta ** 2 / ((alpha - 1) ** 2 * (alpha - 2))
            return (mean - mean_guess) ** 2 + (sd ** 2 - var_guess) ** 2

        params = _get_shape_scale(pd.DataFrame(exposure['mean']), pd.DataFrame(exposure['standard_deviation']), f,
                                  lambda m, s: np.array((m, m * s)))
        try:
            shape, scale = np.abs(params['shape']), np.abs(params['scale'])
            return {'a': shape, 'scale': scale}

        except TypeError:
            print('InverseGamma did not converge!!')


class InverseWeibull(BaseDistribution):
    def _get_params(self, exposure):
        # moments from  Stat Papers (2011) 52: 591. https://doi.org/10.1007/s00362-009-0271-3
        # it is much faster than using stats.invweibull.mean/var
        def f(guess, mean, sd):
            shape, scale = np.abs(guess)
            mean_guess = scale * special.gamma(1 - 1 / shape)
            var_guess = scale ** 2 * special.gamma(1 - 2 / shape) - mean_guess ** 2
            return (mean - mean_guess) ** 2 + (sd ** 2 - var_guess) ** 2

        params = _get_shape_scale(pd.DataFrame(exposure['mean']), pd.DataFrame(exposure['standard_deviation']), f,
                                  lambda m, s: np.array((max(2.2, s / m), m)))
        try:
            shape, scale = np.abs(params['shape']), np.abs(params['scale'])
            return {'c': shape, 'scale': scale}

        except TypeError:
            print('InverseWeibull did not converge!!')


class LogLogistic(BaseDistribution):
    def _get_params(self, exposure):
        def f(guess, mean, sd):
            shape, scale = np.abs(guess)
            b = np.pi / shape
            mean_guess = scale * b / np.sin(b)
            var_guess = scale ** 2 * 2 * b / np.sin(2 * b) - mean_guess ** 2
            return (mean - mean_guess) ** 2 + (sd ** 2 - var_guess) ** 2

        params = _get_shape_scale(pd.DataFrame(exposure['mean']), pd.DataFrame(exposure['standard_deviation']), f,
                                    lambda m, s: np.array((max(2, m), m)))
        try:
            scale, shape = np.abs(params['scale']), np.abs(params['shape'])
            return {'c': shape,
                    'd': pd.DataFrame([1]*len(exposure), index=exposure.index),
                    'scale': scale}
        except TypeError:
            print('LogLogistic did not converge!!')


class LogNormal(BaseDistribution):
    def _get_params(self, exposure):
        mean, sd = exposure['mean'], exposure['standard_deviation']
        alpha = 1 + sd ** 2 / mean ** 2
        s = np.sqrt(np.log(alpha))
        scale = mean / np.sqrt(alpha)
        return {'s': pd.DataFrame(s, index=exposure.index),
                'scale': pd.DataFrame(scale, index=exposure.index)}


class MirroredGumbel(BaseDistribution):
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
            return 1 - data
        elif process_type == 'ppf_postprocess':
            return ranges['x_max'] - data
        else:
            return data


class MirroredGamma(BaseDistribution):
    def _get_params(self, exposure):
        mean, sd = exposure['mean'], exposure['standard_deviation']
        a = ((self._range['x_max'] - mean) / sd) ** 2
        scale = sd ** 2 / (self._range['x_max'] - mean)
        return {'a': pd.DataFrame(a, index=exposure.index), 'scale': pd.DataFrame(scale, index=exposure.index)}

    def process(self, data, process_type, ranges):
        if process_type == 'pdf_preprocess':
            return ranges['x_max'] - data
        elif process_type == 'ppf_preprocess':
            return 1 - data
        elif process_type == 'ppf_postprocess':
            return ranges['x_max'] - data
        else:
            return data


class Normal(BaseDistribution):
    def _get_params(self, exposure):
        return {'loc': pd.DataFrame(exposure['mean'], index=exposure.index),
                'scale': pd.DataFrame(exposure['standard_deviation'], index=exposure.index)}


class Weibull(BaseDistribution):
    def _get_params(self, exposure):
        def f(guess, mean, sd):
            shape, scale = np.abs(guess)
            mean_guess = scale * special.gamma(1 + 1 / shape)
            var_guess = scale ** 2 * special.gamma(1 + 2 / shape) - mean_guess ** 2
            return (mean - mean_guess) ** 2 + (sd ** 2 - var_guess) ** 2

        params = _get_shape_scale(pd.DataFrame(exposure['mean']), pd.DataFrame(exposure['standard_deviation']), f,
                                  lambda m, s: np.array((m, m / s)))
        try:
            shape, scale = np.abs(params['shape']), np.abs(params['scale'])
            return {'c': shape, 'scale': scale}

        except TypeError:
            print('Weibull did not converge!!')

class EnsembleDistribution:

    distribution_map = {'betasr': (Beta, stats.beta),
                        'exp': (Exponential, stats.expon),
                        'gamma': (Gamma, stats.gamma),
                        'gumbel': (Gumbel, stats.gumbel_r),
                        'invgamma': (InverseGamma, stats.invgamma),
                        'invweibull': (InverseWeibull, stats.invweibull),
                        'llogis': (LogLogistic, stats.burr12),
                        'lnorm': (LogNormal, stats.lognorm),
                        'mgamma': (MirroredGamma, stats.gamma),
                        'mgumbel': (MirroredGumbel, stats.gumbel_r),
                        'norm': (Normal, stats.norm),
                        'weibull': (Weibull, stats.weibull_min)}

    def __init__(self, exposure, weights):
        self._distributions, self.weights = self.get_valid_distributions(self.distribution_map, exposure, weights)

    @staticmethod
    def get_valid_distributions(maps, exposure, weights):
        weights = weights.loc[:, 'exp':'mgumbel']
        if 'glnorm' in weights:
            weights = weights.drop('glnorm', axis=1)

        # weight is all same across the demo groups
        e_weights = weights.iloc[0]

        # make sure that e_weights are properly scaled
        e_weights = e_weights / np.sum(e_weights)
        dist = dict()

        # we drop the invweibull if its weight is less than 5 percent
        if 'invweibull' in e_weights and e_weights['invweibull'] < 0.05:
            e_weights = e_weights.drop('invweibull')
            e_weights = e_weights / np.sum(e_weights)

        for dist_name in e_weights.index:
            dist[dist_name] = maps[dist_name][0](exposure, maps[dist_name][1])

        return dist, e_weights / np.sum(e_weights)

    def setup(self, builder):
        builder.components.add_components([self._distributions[dist_name] for dist_name in self._distributions.keys()])

    def pdf(self, x):
        return np.sum([self.weights[dist_name] * self._distributions[dist_name].pdf(x) for dist_name in
                       self._distributions.keys()])

    def ppf(self, x):
        return np.sum([self.weights[dist_name] * self._distributions[dist_name].ppf(x) for dist_name in
                       self._distributions.keys()], axis=0)

    def pdf_test(self, x):
        with np.errstate(all='warn'):
            return np.sum([self.weights[dist_name] * self._distributions[dist_name].pdf_test(x) for dist_name in
                           self._distributions.keys()], axis=0)

    def ppf_test(self, x):
        with np.errstate(all='warn'):
            return np.sum([self.weights[dist_name] * self._distributions[dist_name].ppf_test(x) for dist_name in
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
        if risk == 'high_ldl_cholesterol':
            weights = weights.drop('invgamma', axis=1)
        return EnsembleDistribution(exposure, weights)

    # For 2016, we don't have any lognormal/normal risk factor with actual data
    elif distribution == 'lognormal':
        return LogNormal(exposure, stats.lognorm)
    elif distribution == 'normal':
        return Normal(exposure, stats.norm)
    else:
        raise ValueError(f"Unhandled distribution type {distribution}")
