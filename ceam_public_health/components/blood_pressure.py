import os.path
from functools import partial

import pandas as pd
import numpy as np
from scipy.stats import norm

from ceam import config

from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns
from ceam.framework.values import modifies_value

from ceam_inputs.gbd_ms_functions import load_data_from_cache, normalize_for_simulation, get_sbp_mean_sd
from ceam_inputs import make_gbd_risk_effects
from ceam_inputs.util import gbd_year_range

from ceam_public_health.util.risk import continuous_exposure_effect

class BloodPressure:
    """
    Model systolic blood pressure and it's effect on IHD and stroke

    Population Columns
    ------------------
    systolic_blood_pressure_percentile
        Each simulant's position in the population level SBP distribution. A simulant with .99 will always have high blood pressure and a simulant with .01 will always be low relative to the current average
    systolic_blood_pressure
        Each simulant's current SBP
    """

    def setup(self, builder):
        self.sbp_distribution = builder.lookup(self.load_sbp_distribution())
        self.randomness = builder.randomness('blood_pressure')

        effect_function = continuous_exposure_effect('systolic_blood_pressure', tmrl=112.5, scale=10)
        risk_effects = make_gbd_risk_effects(107, [
            (493, 'heart_attack'),
            (496, 'hemorrhagic_stroke'),
            (495, 'ischemic_stroke'),
            (591, 'ckd'),
            ], effect_function)

        return risk_effects

    @listens_for('initialize_simulants')
    @uses_columns(['systolic_blood_pressure_percentile', 'systolic_blood_pressure'])
    def load_population_columns(self, event):
        population_size = len(event.index)
        event.population_view.update(pd.DataFrame({
            'systolic_blood_pressure_percentile': self.randomness.get_draw(event.index)*0.98+0.01,
            'systolic_blood_pressure': np.full(population_size, 112.0),
            }))

    def load_sbp_distribution(self):
        location_id = config.getint('simulation_parameters', 'location_id')
        year_start, year_end = gbd_year_range()

        distribution = load_data_from_cache(get_sbp_mean_sd, col_name=['log_mean', 'log_sd'],
                            src_column=['log_mean_{draw}', 'log_sd_{draw}'],
                            location_id=location_id, year_start=year_start, year_end=year_end)


        return distribution

    @listens_for('time_step__prepare', priority=8)
    @uses_columns(['systolic_blood_pressure', 'systolic_blood_pressure_percentile'], 'alive')
    def update_systolic_blood_pressure(self, event):
        distribution = self.sbp_distribution(event.index)
        new_sbp = np.exp(norm.ppf(event.population.systolic_blood_pressure_percentile,
                                  loc=distribution['log_mean'], scale=distribution['log_sd']))
        event.population_view.update(pd.Series(new_sbp, name='systolic_blood_pressure', index=event.index))
