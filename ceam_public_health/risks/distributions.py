import numpy as np
import pandas as pd
from scipy import stats, optimize, integrate, special

from vivarium.interpolation import Interpolation


def get_min_max(exposure_mean, exposure_sd):
    # Construct parameters for a lognormal distribution
    exposure = exposure_mean.merge(exposure_sd).set_index(['age', 'sex', 'year'])

    alpha = 1 + exposure['standard_deviation'].values**2/exposure['mean'].values**2
    scale = exposure['mean'].values/np.sqrt(alpha)
    s = np.sqrt(np.log(alpha))
    x_min = stats.lognorm(s=s, scale=scale).ppf([.001] * len(exposure))
    x_max = stats.lognorm(s=s, scale=scale).ppf([.999] * len(exposure))

    x_min = pd.DataFrame({'x_min': x_min}, index=exposure.index).reset_index()
    x_max = pd.DataFrame({'x_max': x_max}, index=exposure.index).reset_index()

    return x_min, x_max


class BaseDistribution:
    def __init__(self, risk, risk_type="risk_factor", weights=None):
        self._risk = risk
        self._risk_type = risk_type
        self._weights = weights
        self._params = None

    @property
    def params(self):
        if self._params is None:
            self._params = self.get_parameters()
        return self._params

    def get_parameters(self):
        raise NotImplementedError()

    def setup(self, builder):
        self._exposure_mean = builder.data.load(f"{self._risk_type}.{self._risk}.exposure")
        self._exposure_sd = builder.data.load(f"{self._risk_type}.{self._risk}.exposure_standard_deviation")
        #self._exposure_data = exposure_data.merge(exposure_sd_data).set_index(['age', 'sex', 'year'])

class Beta(BaseDistribution):
    def get_parameters(self):
        x_max, x_min = get_min_max(self._exposure_mean, self._exposure_sd)
        scale = (x_max - x_min)
        a = 1 / scale * (self._exposure_mean - x_min)
        b = (1 / scale * self._exposure_sd) ** 2
        shape_1 = a ** 2 / b * (1 - a) - a
        shape_2 = a / b * (1 - a) ** 2 + (a - 1)
        return scale, shape_1, shape_2, x_min

    def pdf(self, x):
        scale, shape_1, shape_2, x_min = self.params
        y = (x - x_min)
        return stats.beta(a=shape_1, b=shape_2, scale=scale).pdf(y)

    def ppf(self, x):
        return stats.beta(a=shape_1, b=shape_2, scale=scale).ppf(x)


class Exponential(BaseDistribution):
    def get_parameters(self):
        return self._exposure_mean

    def pdf(self, x):
        scale = self.params
        return stats.expon(scale=scale).pdf(x)

    def ppf(self, x):
        scale = self.params
        return stats.expon(scale=scale).ppf(x)


class Gamma(BaseDistribution):
    def get_parameters(self):
        a = (self._exposure_mean / self._exposure_sd) ** 2
        scale = self._exposure_sd ** 2 / self._exposure_mean
        return a, scale

    def pdf(self, x):
        a, scale = self.params
        return stats.gamma(a=a, scale=scale).pdf(x)

    def ppf(self, x):
        a, scale = self.params
        return stats.gamma(a=a, scale=scale).ppf(x)


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
    def get_parameters(self):
        loc = self._exposure_mean - (np.euler_gamma * np.sqrt(6) / np.pi * self._exposure_sd)
        scale = np.sqrt(6) / np.pi * self._exposure_sd
        return loc, scale

    def pdf(self, x):
        loc, scale = self.params
        return stats.gumbel_r(loc=loc, scale=scale).pdf(x)

    def ppf(self, x):
        loc, scale = self.params
        return stats.gumbel_r(loc=loc, scale=scale).ppf(x)


class InverseGamma(BaseDistribution):
    def get_parameters(self):
        def f(guess):
            alpha, beta = np.abs(guess)
            mean_guess = beta / (alpha - 1)
            var_guess = beta ** 2 / ((alpha - 1) ** 2 * (alpha - 2))
            return (self._exposure_mean - mean_guess) ** 2 + (self._exposure_sd ** 2 - var_guess) ** 2

        initial_guess = np.array((self._exposure_mean, self._exposure_mean * self._exposure_sd))
        result = optimize.minimize(f, initial_guess, method='Nelder-Mead')
        assert result.success

        return result.x

    def pdf(self, x):
        a, scale = self.params
        return stats.invgamma(a=a, scale=scale).pdf(x)

    def ppf(self, x):
        a, scale = self.params
        return stats.invgamma(a=a, scale=scale).ppf(x)


class InverseWeibull(BaseDistribution):

    def __init__(self, exposure_mean, exposure_sd, x_min, x_max):
        self.c, self.scale = self._get_params(exposure_mean, exposure_sd, x_min, x_max)

    @staticmethod
    def _get_params(exposure_mean, exposure_sd, _, __):

        def x_inverse_weibull(x, shape, scale):
            return x * stats.invweibull.pdf(x, c=shape, scale=scale)

        def x2_inverse_weibull(x, shape, scale):
            return x ** 2 * stats.invweibull.pdf(x, c=shape, scale=scale)

        def f(guess):
            mean_guess = integrate.quad(x_inverse_weibull, 0, np.inf, *guess, epsrel=0.1, epsabs=0.1)[0]
            param_guess = integrate.quad(x2_inverse_weibull, 0, np.inf, *guess, epsrel=0.1, epsabs=0.1)[0]
            var_guess = param_guess - mean_guess ** 2
            return (exposure_mean - mean_guess) ** 2 + (exposure_sd ** 2 - var_guess) ** 2

        initial_guess = np.array((max(2.2, exposure_sd / exposure_mean), exposure_mean))
        result = optimize.minimize(f, initial_guess, method='Nelder-Mead')
        assert result.success

        return result.x

    def pdf(self, x):
        return stats.invweibull(c=self.c, scale=self.scale).pdf(x)

    def ppf(self, x):
        return stats.invweibull(c=self.c, scale=self.scale).ppf(x)


class LogLogistic(BaseDistribution):
    def get_parameters(self):
        def f(guess):
            a, b = np.abs((guess[0], np.pi / guess[1]))
            mean_guess = a * b / np.sin(b)
            var_guess = a ** 2 * (2 * b / np.sin(2 * b)) - (b ** 2 / np.sin(b) ** 2)
            return (self._exposure_mean - mean_guess) ** 2 + (self._exposure_sd ** 2 - var_guess) ** 2

        initial_guess = np.array((self._exposure_mean, max(2, self._exposure_mean)))
        result = optimize.minimize(f, initial_guess, method='Nelder-Mead')
        assert result.success

        return result.x

    def pdf(self, x):
        scale, c = self.params
        return stats.fisk(c=c, scale=scale).pdf(x)

    def ppf(self, x):
        scale, c = self.params
        return stats.fisk(c=c, scale=scale).ppf(x)


class LogNormal(BaseDistribution):
    def setup(self, builder):
        super().setup(builder)
        #FIXME: There should be a better way to defer this
        self._build_lookup_function = builder.lookup

    def get_parameters(self):
        exposure_mean, exposure_sd = self._exposure_mean.values, self._exposure_sd.values
        alpha = 1 + exposure_sd ** 2 / exposure_mean ** 2
        s = np.sqrt(np.log(alpha))
        scale = exposure_mean / np.sqrt(alpha)
        parameters = pd.DataFrame({'s': s, 'scale': scale}, index=self._exposure_mean).reset_index()
        return self._build_lookup_function(parameters)

    def pdf(self, x):
        params = self.params(x.index)
        return stats.lognorm(s=params['s'], scale=params['scale']).pdf(x)

    def ppf(self, propensity):
        params = self.params(propensity.index)
        return stats.lognorm(s=params['s'], scale=params['scale']).ppf(propensity)


class MirroredGamma(BaseDistribution):
    def get_parameters(self):
        x_max, x_min = get_min_max(self._exposure_mean, self._exposure_sd)
        a = ((x_max - self._exposure_mean) / self._exposure_sd) ** 2
        scale = self._exposure_sd ** 2 / (x_max - self._exposure_mean)
        return a, scale, x_max

    def pdf(self, x):
        a, scale, x_max = self.params
        y = x_max - x
        return stats.gamma(a=a, scale=scale).pdf(y)

    def ppf(self, x):
        a, scale, _ = self.params
        return stats.gamma(a=a, scale=scale).ppf(x)


class MirroredGumbel(BaseDistribution):
    def get_parameters(self):
        x_max, x_min = get_min_max(self._exposure_mean, self._exposure_sd)
        loc = x_max - self._exposure_mean - (np.euler_gamma * np.sqrt(6) / np.pi * self._exposure_sd)
        scale = np.sqrt(6) / np.pi * self._exposure_sd
        return loc, scale, x_max

    def pdf(self, x):
        loc, scale, x_max = self.params
        y = x_max - x
        return stats.gumbel_r(loc=loc, scale=scale).pdf(y)

    def ppf(self, x):
        loc, scale, _ = self.params
        return stats.gumbel_r(loc=loc, scale=scale).ppf(x)


class Normal(BaseDistribution):
    def setup(self, builder):
        super().setup(builder)
        #FIXME: There should be a better way to defer this
        self._build_lookup_function = builder.lookup

    def get_parameters(self):
        dist = self._exposure_sd.merge(self._exposure_mean, on=['year', 'sex', 'age'])
        dist = dist.rename(columns={'value_x': 'scale', 'value_y': 'loc'})
        return self._build_lookup_function(dist[["year", "sex", "age", "loc", "scale"]])

    def pdf(self, x):
        params = self.params(x.index)
        return stats.norm(loc=params['loc'], scale=params['scale']).pdf(x)

    def ppf(self, propensity):
        params = self.params(propensity.index)
        return stats.norm(loc=params['loc'], scale=params['scale']).ppf(propensity)


class Weibull(BaseDistribution):
    def get_parameters(self):
        def f(guess):
            scale, shape = guess
            mean_guess = scale * special.gamma(1 + 1 / shape)
            var_guess = scale ** 2 * special.gamma(1 + 2 / shape) - special.gamma(1 + 1 / shape) ** 2
            return (self._exposure_mean - mean_guess) ** 2 + (self._exposure_sd ** 2 - var_guess) ** 2

        initial_guess = np.array((self._exposure_mean / self._exposure_sd, self._exposure_mean))
        result = optimize.minimize(f, initial_guess, method='Nelder-Mead')
        assert result.success

        return result.x

    def pdf(self, x):
        scale, c = self.params
        return stats.weibull_min(c=c, scale=scale).pdf(x)

    def ppf(self, x):
        scale, c = self.params
        return stats.weibull_min(c=c, scale=scale).ppf(x)


# FIXME: several of the distributions do not currently work
class EnsembleDistribution:

    distribution_map = {'betasr': Beta,
                        'exp': Exponential,
                        'gamma': Gamma,
                        'glnorm': GeneralizedLogNormal,
                        'gumbel': Gumbel,
                        'invgamma': InverseGamma,
                        'invweibull': InverseWeibull,
                        'llogis': LogLogistic,
                        'lnorm': LogNormal,
                        'mgamma': MirroredGamma,
                        'mgumbel': MirroredGumbel,
                        'norm': Normal,
                        'weibull': Weibull}

    def __init__(self, risk, risk_type, weights):
        self._distribution = Normal(risk, risk_type, weights=weights)

        # self._distributions = {distribution_name: distribution(exposure_mean, exposure_sd, x_min, x_max)
        #                        for distribution_name, distribution in self.distribution_map}

    def setup(self, builder):
        self._distribution.setup(builder)

    def pdf(self, x):
        return self._distribution.pdf(x)
        #return np.sum([weight * self._distributions[dist_name].pdf(x) for dist_name, weight in self.weights.items()])


    def ppf(self, propensity):
        return self._distribution.ppf(propensity)
        #return np.sum([weight * self._distributions[dist_name].ppf(x) for dist_name, weight in self.weights.items()])


def get_distribution(risk, risk_type, builder, weights=None):
    distribution = builder.data.load(f"{risk_type}.{risk}.distribution")
    if distribution == 'ensemble':
        return EnsembleDistribution(risk, risk_type, weights)
    elif distribution == 'lognormal':
        return LogNormal(risk, risk_type, exposure)
    elif distribution == 'normal':
        return Normal(risk, risk_type, exposure)
    else:
        raise ValueError(f"Unhandled distribution type {distribution}")
