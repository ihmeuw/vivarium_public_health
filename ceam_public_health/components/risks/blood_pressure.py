import numpy as np

from scipy.stats import norm

from ceam import config

from ceam_inputs.gbd_ms_functions import load_data_from_cache, get_sbp_mean_sd
from ceam_inputs.util import gbd_year_range

def distribution_loader(builder):
    location_id = config.simulation_parameters.location_id
    year_start, year_end = gbd_year_range()

    distribution = load_data_from_cache(get_sbp_mean_sd, col_name=['log_mean', 'log_sd'],
                        src_column=['log_mean_{draw}', 'log_sd_{draw}'],
                        location_id=location_id, year_start=year_start, year_end=year_end)


    return builder.lookup(distribution)

def exposure_function(propensity, distribution):
    return np.exp(norm.ppf(propensity, loc=distribution['log_mean'], scale=distribution['log_sd']))
