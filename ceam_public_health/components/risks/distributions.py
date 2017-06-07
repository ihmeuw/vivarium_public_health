import numpy as np
import pandas as pd
from scipy.stats import norm, beta

from ceam.interpolation import Interpolation

from ceam_inputs import (get_fpg_distribution_parameters, get_bmi_distribution_parameters,
                         get_sbp_distribution, get_exposures, risk_factors)


def _sll_ppf(percentile, location, scale, shape):
    """ compute the value of the shifted-log-logistic distribution
    Parameters
    ----------
    percentile : float or array of floats between 0 and 1
    location, scale, shape : floats or array of floats, scale > 0

    Returns
    -------
    returns float or array of floats
    """
    assert np.all(scale > 0), 'scale must be positive'

    percentile = np.atleast_1d(percentile)
    location = np.broadcast_to(location, percentile.shape)
    scale = np.broadcast_to(scale, percentile.shape)
    shape = np.broadcast_to(shape, percentile.shape)

    f = 1. - percentile
    idx = f != 0

    z = 1/shape[idx] * ((1/f[idx] - 1)**shape[idx] - 1)
    x = location[idx] + scale[idx]*z

    result = np.full(f.shape, np.inf)
    result[idx] = x

    if len(result) > 1:
        return result
    else:
        return result[0]


def _fpg_ppf(parameters):
    def inner(percentile):
        if parameters.empty:
            return pd.Series()
        else:
            return _sll_ppf(percentile, parameters['loc'], parameters['scale'], parameters['error'])
    return inner


def _bmi_ppf(parameters):
    return beta(a=parameters['a'], b=parameters['b'], scale=parameters['scale'], loc=parameters['loc']).ppf


def fpg(builder):
    distribution = Interpolation(get_fpg_distribution_parameters(),
                                 categorical_parameters=('sex', 'location'),
                                 continuous_parameters=('age', 'year'),
                                 func=_fpg_ppf)
    return builder.lookup(distribution, key_columns=('sex', 'location'))


def sbp(builder):
    return builder.lookup(get_sbp_distribution())


def cholesterol(builder):
    df = get_exposures(risk_factors.high_total_cholesterol.gbd_risk)
    # NOTE: Cholesterol is not modeled for younger ages so set them equal to the TMRL
    df.loc[df.age < 27.5, 'continuous'] = 3.08
    df = df.set_index(['age', 'sex', 'year'])
    means = df.mean(axis=1)
    means.name = 'mean'
    std = np.sqrt(means)
    std.name = 'std'
    dist = pd.concat([means, std], axis=1).reset_index()
    dist = Interpolation(dist, ['sex'], ['age', 'year'],
                         func=lambda parameters: norm(loc=parameters['mean'], scale=parameters['std']).ppf)
    return builder.lookup(dist)


def bmi(builder):
    distibution = Interpolation(get_bmi_distribution_parameters(),
                                categorical_parameters=('sex',),
                                continuous_parameters=('age', 'year'),
                                func=_bmi_ppf)
    return builder.lookup(distibution)

