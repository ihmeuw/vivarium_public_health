import pandas as pd
import numpy as np

from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns

from ceam_inputs import make_gbd_risk_effects
from ceam_inputs import get_fpg_distributions

from ceam_public_health.util.risk import continuous_exposure_effect

class FastingPlasmaGlucose:
    """Model FastingPlasmaGlucose

    Population Columns
    ------------------
    fpg_percentile
        Position of the simulant in the population's fasting plasma glucose distribution
    """

    def setup(self, builder):
        self.fpg_distributions = builder.lookup(get_fpg_distributions())
        self.randomness = builder.randomness('fpg')

        effect_function = continuous_exposure_effect('fpg', tmrl=5.1, scale=1)
        risk_effects = make_gbd_risk_effects(141, [
            (493, 'heart_attack'),
            (496, 'hemorrhagic_stroke'),
            (495, 'ischemic_stroke'),
            ], effect_function)
        return risk_effects


    @listens_for('initialize_simulants')
    @uses_columns(['fpg_percentile', 'fpg'])
    def initialize(self, event):
        event.population_view.update(pd.DataFrame({
            'fpg_percentile': self.randomness.get_draw(event.index)*0.98+0.01,
            'fpg': np.full(len(event.index), 20)
        }))

    @listens_for('time_step__prepare', priority=8)
    @uses_columns(['fpg', 'fpg_percentile'], 'alive')
    def update_fasting_plasma_glucose(self, event):
        new_fpg = self.fpg_distributions(event.index)(event.population.fpg_percentile)
        event.population_view.update(pd.Series(new_fpg, name='fpg', index=event.index))

