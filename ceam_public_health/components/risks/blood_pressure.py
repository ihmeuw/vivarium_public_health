import numpy as np
from scipy.stats import norm

from ceam_inputs import get_sbp_distribution


def distribution_loader(builder):
    return builder.lookup(get_sbp_distribution())


def exposure_function(propensity, distribution):
    return np.exp(norm.ppf(propensity, loc=distribution['log_mean'], scale=distribution['log_sd']))
