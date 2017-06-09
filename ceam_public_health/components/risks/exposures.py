import numpy as np
from scipy.stats import norm


def basic_exposure_function(propensity, distribution):
    """This function handles the simple common case for getting a simulant's
    based on their propensity for the risk. Some risks will require a more
    complex version of this.

    Parameters
    ----------
    propensity : pandas.Series
        The propensity for each simulant
    distribution : callable
        A function with maps propensities to values from the distribution
    """
    return distribution(propensity)


def sbp(propensity, distribution):
    return np.exp(norm.ppf(propensity, loc=distribution['log_mean'], scale=distribution['log_sd']))
