import numpy as np
from scipy import stats, optimize, integrate, special

def get_min_max(exposure_mean, exposure_sd):
    # Construct parameters for a lognormal distribution
    alpha = 1 + exposure_sd**2/exposure_mean**2
    scale = exposure_mean/np.sqrt(alpha)
    s = np.sqrt(np.log(alpha))

    x_min, x_max = stats.lognorm.ppf([0.001, 0.999], s=s, scale=scale)

    return x_min, x_max


def ensemble_beta(exposure_mean, exposure_sd, x_min, x_max):
    scale = (x_max - x_min)
    a = 1 / scale * (exposure_mean - x_min)
    b = (1 / scale * exposure_sd)**2

    shape_1 = a**2/b * (1 - a) - a
    shape_2 = a/b * (1 - a)**2 + (a - 1)

    def beta_ppf(x):
        y = 1 / scale * (x - x_min)
        return 1 / scale * stats.beta(shape_1, shape_2).ppf(y)

    return beta_ppf


def ensemble_exponential(exposure_mean, _, __, ___):
    rate = 1/exposure_mean

    def exponential_ppf(x):
        y = rate * x
        return rate * stats.expon().ppf(y)

    return exponential_ppf


def ensemble_gamma(exposure_mean, exposure_sd, _, __):
    a = (exposure_mean/exposure_sd)**2
    scale = exposure_sd**2/exposure_mean

    def gamma_ppf(x):
        y = x / scale
        return 1 / scale * stats.gamma(a).ppf(y)

    return gamma_ppf


def ensemble_generalized_log_normal(*_):
    """Here for completeness.

    The weight for the glnorm distribution is zero for all ensemble modeled risks for 2016.  Which is good, because
    getting the parameters involves running some optimization technique, and I definitely don't want to spend
    time ensuring R's optimization scheme is equivalent to scipy's.  - J.C.
    """
    def generalized_log_normal_ppf(_):
        return 0

    return generalized_log_normal_ppf


def ensemble_gumbel(exposure_mean, exposure_sd, _, __):
    alpha = exposure_mean - (np.euler_gamma*np.sqrt(6)/np.pi * exposure_sd)
    scale = np.sqrt(6)/np.pi * exposure_sd

    def gumbel_ppf(x):
        y = (x - alpha)/scale
        return 1 / scale * stats.gumbel_r().ppf(y)

    return gumbel_ppf


def ensemble_inverse_gamma(exposure_mean, exposure_sd, _, __):

    def f(guess):
        alpha, beta = guess
        mean_guess = beta/(alpha - 1)
        var_guess = beta**2 / ((alpha - 1)**2 * (alpha - 2))
        return (exposure_mean - mean_guess)**2 + (exposure_sd**2 - var_guess)**2

    initial_guess = np.array((exposure_mean, exposure_mean*exposure_sd))
    result = optimize.minimize(f, initial_guess, method='Nelder-Mead')
    assert result.success

    a, scale = result.x

    def inverse_gamma_ppf(x):
        y = x/scale
        return 1 / scale * stats.invgamma(a).ppf(y)

    return inverse_gamma_ppf


def ensemble_inverse_weibull(exposure_mean, exposure_sd, _, __):

    def x_inverse_weibull(x, shape, scale):
        return x * stats.invweibull.pdf(x, c=shape, scale=scale)

    def x2_inverse_weibull(x, shape, scale):
        return x**2 * stats.invweibull.pdf(x, c=shape, scale=scale)

    def f(guess):
        mean_guess = integrate.quad(x_inverse_weibull, 0, np.inf, *guess, epsrel=0.1, epsabs=0.1)[0]
        param_guess = integrate.quad(x2_inverse_weibull, 0, np.inf, *guess, epsrel=0.1, epsabs=0.1)[0]
        var_guess = param_guess - mean_guess**2
        return (exposure_mean - mean_guess)**2 + (exposure_sd**2 - var_guess)**2

    initial_guess = np.array((max(2.2, exposure_sd/exposure_mean), exposure_mean))
    result = optimize.minimize(f, initial_guess, method='Nelder-Mead')
    assert result.success

    c, scale = result.x

    def inverse_weibull_ppf(x):
        y = x/scale
        return 1 / scale * stats.invweibull(c).ppf(y)

    return inverse_weibull_ppf


def ensemble_log_logistic(exposure_mean, exposure_sd, _, __):

    def f(guess):
        a, b = guess[0], np.pi/guess[1]
        mean_guess = a * b / np.sin(b)
        var_guess = a**2 * (2 * b / np.sin(2*b)) - (b**2 / np.sin(b)**2)
        return (exposure_mean - mean_guess)**2 + (exposure_sd**2 - var_guess)**2

    initial_guess = np.array((exposure_mean, max(2, exposure_mean)))
    result = optimize.minimize(f, initial_guess, method='Nelder-Mead')
    assert result.success

    scale, c = result.x

    def log_logistic_ppf(x):
        y = x/scale
        return 1/scale * stats.fisk(c).ppf(y)

    return log_logistic_ppf


def ensemble_log_normal(exposure_mean, exposure_sd, _, __):
    alpha = 1 + exposure_sd**2 / exposure_mean**2
    s = np.sqrt(np.log(alpha))
    scale = exposure_mean/np.sqrt(alpha)

    def log_normal_ppf(x):
        y = x/scale
        return 1/scale * stats.lognorm(s).ppf(y)

    return log_normal_ppf


def ensemble_mirrored_gamma(exposure_mean, exposure_sd, _, x_max):
    a = ((x_max - exposure_mean)/exposure_sd)**2
    scale = exposure_sd**2/(x_max - exposure_mean)

    def mirrored_gamma_ppf(x):
        y = (x_max - x)/scale
        return 1 / scale * stats.gamma(a).ppf(y)

    return mirrored_gamma_ppf


def ensemble_mirrored_gumbel(exposure_mean, exposure_sd, _, x_max):
    alpha = x_max - exposure_mean - (np.euler_gamma * np.sqrt(6) / np.pi * exposure_sd)
    scale = np.sqrt(6) / np.pi * exposure_sd

    def mirrored_gumbel_ppf(x):
        y = (x_max - x - alpha)/scale
        return 1 / scale * stats.gumbel_r().ppf(y)

    return mirrored_gumbel_ppf


def ensemble_normal(exposure_mean, exposure_sd, _, __):

    def normal_ppf(x):
        return stats.norm(exposure_mean, exposure_sd).ppf(x)

    return normal_ppf


def ensemble_weibull(exposure_mean, exposure_sd, _, __):

    def f(guess):
        scale, shape = guess
        mean_guess = scale * special.gamma(1 + 1/shape)
        var_guess = scale**2 * special.gamma(1 + 2/shape) - special.gamma(1 + 1/shape)**2
        return (exposure_mean - mean_guess)**2 + (exposure_sd**2 - var_guess)**2

    initial_guess = np.array((exposure_mean/exposure_sd, exposure_mean))
    result = optimize.minimize(f, initial_guess, method='Nelder-Mead')
    assert result.success

    scale, c = result.x

    def weibull_ppf(x):
        y = x / scale
        return 1 / scale * stats.weibull_min(c).ppf(y)

    return weibull_ppf


def ensemble_distribution(weights, exposure_mean, exposure_sd):
    x_min, x_max = get_min_max(exposure_mean, exposure_sd)

    distribution_function_map = {'betasr': ensemble_beta,
                                 'exp': ensemble_exponential,
                                 'gamma': ensemble_gamma,
                                 'glnorm': ensemble_generalized_log_normal,
                                 'gumbel': ensemble_gumbel,
                                 'invgamma': ensemble_inverse_gamma,
                                 'invweibull': ensemble_inverse_weibull,
                                 'llogis': ensemble_log_logistic,
                                 'lnorm': ensemble_log_normal,
                                 'mgamma': ensemble_mirrored_gamma,
                                 'mgumbel': ensemble_mirrored_gumbel,
                                 'norm': ensemble_normal,
                                 'weibull': ensemble_weibull}

    distribution_ppf = {distribution_name: distribution(exposure_mean, exposure_sd, x_min, x_max)
                        for distribution_name, distribution in distribution_function_map}

    def ensemble_ppf(x):
        np.sum([weight*distribution_ppf[dist](x) for weight, dist in weights.items()])

    return ensemble_ppf
