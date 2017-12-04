import numpy as np
from scipy import stats, optimize, integrate, special


def get_min_max(exposure_mean, exposure_sd):
    # Construct parameters for a lognormal distribution
    alpha = 1 + exposure_sd**2/exposure_mean**2
    scale = exposure_mean/np.sqrt(alpha)
    s = np.sqrt(np.log(alpha))

    x_min, x_max = stats.lognorm.ppf([0.001, 0.999], s=s, scale=scale)

    return x_min, x_max


class Beta:

    def __init__(self, exposure_mean, exposure_sd, x_min, x_max):
        self.scale, self.shape_1, self.shape_2 = self._get_params(exposure_mean, exposure_sd, x_min, x_max)
        self.x_min = x_min

    @staticmethod
    def _get_params(exposure_mean, exposure_sd, x_min, x_max):
        scale = (x_max - x_min)
        a = 1 / scale * (exposure_mean - x_min)
        b = (1 / scale * exposure_sd) ** 2
        shape_1 = a ** 2 / b * (1 - a) - a
        shape_2 = a / b * (1 - a) ** 2 + (a - 1)
        return scale, shape_1, shape_2

    def pdf(self, x):
        y = (x - self.x_min)
        return stats.beta(a=self.shape_1, b=self.shape_2, scale=self.scale).pdf(y)

    def ppf(self, x):
        return stats.beta(a=self.shape_1, b=self.shape_2, scale=self.scale).ppf(x)


class Exponential:

    def __init__(self, exposure_mean, exposure_sd, x_min, x_max):
        self.scale = self._get_params(exposure_mean, exposure_sd, x_min, x_max)

    @staticmethod
    def _get_params(exposure_mean, _, __, ___):
        return exposure_mean

    def pdf(self, x):
        return stats.expon(scale=self.scale).pdf(x)

    def ppf(self, x):
        return stats.expon(scale=self.scale).ppf(x)


class Gamma:

    def __init__(self, exposure_mean, exposure_sd, x_min, x_max):
        self.a, self.scale = self._get_params(exposure_mean, exposure_sd, x_min, x_max)

    @staticmethod
    def _get_params(exposure_mean, exposure_sd, _, __):
        a = (exposure_mean / exposure_sd) ** 2
        scale = exposure_sd ** 2 / exposure_mean
        return a, scale

    def pdf(self, x):
        return stats.gamma(a=self.a, scale=self.scale).pdf(x)

    def ppf(self, x):
        return stats.gamma(a=self.a, scale=self.scale).ppf(x)


class GeneralizedLogNormal:
    """Here for completeness.

    The weight for the glnorm distribution is zero for all ensemble modeled risks for 2016.  Which is good, because
    getting the parameters involves running some optimization technique, and I definitely don't want to spend
    time ensuring R's optimization scheme is equivalent to scipy's.  - J.C.
    """
    def __init__(self, *_):
        pass

    def pdf(self, _):
        return 0

    def ppf(self, _):
        return 0


class Gumbel:

    def __init__(self, exposure_mean, exposure_sd, x_min, x_max):
        self.loc, self.scale = self._get_params(exposure_mean, exposure_sd, x_min, x_max)

    @staticmethod
    def _get_params(exposure_mean, exposure_sd, _, __):
        loc = exposure_mean - (np.euler_gamma * np.sqrt(6) / np.pi * exposure_sd)
        scale = np.sqrt(6) / np.pi * exposure_sd
        return loc, scale

    def pdf(self, x):
        return stats.gumbel_r(loc=self.loc, scale=self.scale).pdf(x)

    def ppf(self, x):
        return stats.gumbel_r(loc=self.loc, scale=self.scale).ppf(x)


class InverseGamma:

    def __init__(self, exposure_mean, exposure_sd, x_min, x_max):
        self.a, self.scale = self._get_params(exposure_mean, exposure_sd, x_min, x_max)


    @staticmethod
    def _get_params(exposure_mean, exposure_sd, _, __):

        def f(guess):
            alpha, beta = np.abs(guess)
            mean_guess = beta / (alpha - 1)
            var_guess = beta ** 2 / ((alpha - 1) ** 2 * (alpha - 2))
            return (exposure_mean - mean_guess) ** 2 + (exposure_sd ** 2 - var_guess) ** 2

        initial_guess = np.array((exposure_mean, exposure_mean * exposure_sd))
        result = optimize.minimize(f, initial_guess, method='Nelder-Mead')
        assert result.success

        return result.x

    def pdf(self, x):
        return stats.invgamma(a=self.a, scale=self.scale).pdf(x)

    def ppf(self, x):
        return stats.invgamma(a=self.a, scale=self.scale).ppf(x)


class InverseWeibull:

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


class LogLogistic:

    def __init__(self, exposure_mean, exposure_sd, x_min, x_max):
        self.scale, self.c = self._get_params(exposure_mean, exposure_sd, x_min, x_max)

    @staticmethod
    def _get_params(exposure_mean, exposure_sd, _, __):
        def f(guess):
            a, b = np.abs((guess[0], np.pi / guess[1]))
            mean_guess = a * b / np.sin(b)
            var_guess = a ** 2 * (2 * b / np.sin(2 * b)) - (b ** 2 / np.sin(b) ** 2)
            return (exposure_mean - mean_guess) ** 2 + (exposure_sd ** 2 - var_guess) ** 2

        initial_guess = np.array((exposure_mean, max(2, exposure_mean)))
        result = optimize.minimize(f, initial_guess, method='Nelder-Mead')
        assert result.success

        return result.x

    def pdf(self, x):
        return stats.fisk(c=self.c, scale=self.scale).pdf(x)

    def ppf(self, x):
        return stats.fisk(c=self.c, scale=self.scale).ppf(x)


class LogNormal:

    def __init__(self, exposure_mean, exposure_sd, x_min, x_max):
        self.s, self.scale = self._get_params(exposure_mean, exposure_sd, x_min, x_max)

    @staticmethod
    def _get_params(exposure_mean, exposure_sd, _, __):
        alpha = 1 + exposure_sd ** 2 / exposure_mean ** 2
        s = np.sqrt(np.log(alpha))
        scale = exposure_mean / np.sqrt(alpha)
        return s, scale

    def pdf(self, x):
        return stats.lognorm(s=self.s, scale=self.scale).pdf(x)

    def ppf(self, x):
        return stats.lognorm(s=self.s, scale=self.scale).ppf(x)


class MirroredGamma:

    def __init__(self, exposure_mean, exposure_sd, x_min, x_max):
        self.a, self.scale = self._get_params(exposure_mean, exposure_sd, x_min, x_max)
        self.x_max = x_max

    @staticmethod
    def _get_params(exposure_mean, exposure_sd, _, x_max):
        a = ((x_max - exposure_mean) / exposure_sd) ** 2
        scale = exposure_sd ** 2 / (x_max - exposure_mean)
        return a, scale

    def pdf(self, x):
        y = self.x_max - x
        return stats.gamma(a=self.a, scale=self.scale).pdf(y)

    def ppf(self, x):
        return stats.gamma(a=self.a, scale=self.scale).ppf(x)


class MirroredGumbel:

    def __init__(self, exposure_mean, exposure_sd, x_min, x_max):
        self.loc, self.scale = self._get_params(exposure_mean, exposure_sd, x_min, x_max)
        self.x_max = x_max

    @staticmethod
    def _get_params(exposure_mean, exposure_sd, _, x_max):
        loc = x_max - exposure_mean - (np.euler_gamma * np.sqrt(6) / np.pi * exposure_sd)
        scale = np.sqrt(6) / np.pi * exposure_sd
        return loc, scale

    def pdf(self, x):
        y = self.x_max - x
        return stats.gumbel_r(loc=self.loc, scale=self.scale).pdf(y)

    def ppf(self, x):
        return stats.gumbel_r(loc=self.loc, scale=self.scale).ppf(x)


class Normal:

    def __init__(self, exposure_mean, exposure_sd, x_min, x_max):
        self.loc, self.scale = self._get_params(exposure_mean, exposure_sd, x_min, x_max)

    @staticmethod
    def _get_params(exposure_mean, exposure_sd, _, __):
        return exposure_mean, exposure_sd

    def pdf(self, x):
        return stats.norm(loc=self.loc, scale=self.scale).pdf(x)

    def ppf(self, x):
        return stats.norm(loc=self.loc, scale=self.scale).ppf(x)


class Weibull:

    def __init__(self, exposure_mean, exposure_sd, x_min, x_max):
        self.scale, self.c = self._get_params(exposure_mean, exposure_sd, x_min, x_max)

    @staticmethod
    def _get_params(exposure_mean, exposure_sd, _, __):
        def f(guess):
            scale, shape = guess
            mean_guess = scale * special.gamma(1 + 1 / shape)
            var_guess = scale ** 2 * special.gamma(1 + 2 / shape) - special.gamma(1 + 1 / shape) ** 2
            return (exposure_mean - mean_guess) ** 2 + (exposure_sd ** 2 - var_guess) ** 2

        initial_guess = np.array((exposure_mean / exposure_sd, exposure_mean))
        result = optimize.minimize(f, initial_guess, method='Nelder-Mead')
        assert result.success

        return result.x

    def pdf(self, x):
        return stats.weibull_min(c=self.c, scale=self.scale).pdf(x)

    def ppf(self, x):
        return stats.weibull_min(c=self.c, scale=self.scale).ppf(x)


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

    def __init__(self, exposure_mean, exposure_sd, weights):
        self.weights = weights
        x_min, x_max = get_min_max(exposure_mean, exposure_sd)
        self._distribution = Normal(exposure_mean, exposure_sd, x_min, x_max)

        # self._distributions = {distribution_name: distribution(exposure_mean, exposure_sd, x_min, x_max)
        #                        for distribution_name, distribution in self.distribution_map}

    def pdf(self, x):
        return self._distribution.pdf(x)
        #return np.sum([weight * self._distributions[dist_name].pdf(x) for dist_name, weight in self.weights.items()])


    def ppf(self, x):
        return self._distribution.ppf(x)
        #return np.sum([weight * self._distributions[dist_name].ppf(x) for dist_name, weight in self.weights.items()])


def get_distribution(risk):
    if risk.distribution == 'ensemble':
        return EnsembleDistribution
    elif risk.distribution == 'lognormal':
        return LogNormal
    elif risk.distribution == 'normal':
        return Normal
    else:
        raise ValueError(f"Unhandled distribution type {risk.distribution}")
