import pandas as pd
import numpy as np
from scipy.stats import norm

from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns

from ceam_inputs import get_sbp_distribution, risk_factors

from ceam_public_health.util.risk import continuous_exposure_effect, make_risk_effects


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
        self.default = 112.0
        self.name = 'systolic_blood_pressure'
        self.risk = risk_factors[self.name]
        self.sbp_distribution = builder.lookup(get_sbp_distribution())
        self.randomness = builder.randomness(self.name)

        effect_function = continuous_exposure_effect(self.name, tmrl=self.risk.tmrl, scale=self.risk.scale)
        risk_effects = make_risk_effects(self.risk.gbd_risk,
                                         [(c.gbd_cause, c) for c in self.risk.effected_causes],
                                         effect_function,
                                         self.name)
        return risk_effects

    @listens_for('initialize_simulants')
    @uses_columns(['systolic_blood_pressure_percentile', 'systolic_blood_pressure'])
    def load_population_columns(self, event):
        population_size = len(event.index)
        event.population_view.update(pd.DataFrame({
            '{}_percentile'.format(self.name): self.randomness.get_draw(event.index)*0.98+0.01,
            self.name: np.full(population_size, self.default),
            }))


    @listens_for('time_step__prepare', priority=8)
    @uses_columns(['systolic_blood_pressure', 'systolic_blood_pressure_percentile'], 'alive')
    def update_systolic_blood_pressure(self, event):
        distribution = self.sbp_distribution(event.index)
        new_sbp = np.exp(norm.ppf(event.population.systolic_blood_pressure_percentile,
                                  loc=distribution['log_mean'], scale=distribution['log_sd']))
        event.population_view.update(pd.Series(new_sbp, name='systolic_blood_pressure', index=event.index))
