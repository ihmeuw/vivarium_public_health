import pandas as pd
import numpy as np

from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns

from ceam_inputs import get_bmi_distributions
from ceam_inputs import make_gbd_risk_effects

from ceam_public_health.util.risk import continuous_exposure_effect

class BodyMassIndex:
    """Model BMI

    Population Columns
    ------------------
    bmi_percentile
        Position of the simulant in the population's BMI distribution
    """

    def setup(self, builder):
        self.bmi_distributions = builder.lookup(get_bmi_distributions())
        self.randomness = builder.randomness('bmi')

        effect_function = continuous_exposure_effect('bmi', tmrl=21, scale=5)
        risk_effects = make_gbd_risk_effects(108, [
            (493, 'heart_attack'),
            (496, 'hemorrhagic_stroke'),
            (495, 'ischemic_stroke'),
            ], effect_function, 'bmi')
        return risk_effects


    @listens_for('initialize_simulants')
    @uses_columns(['bmi_percentile', 'bmi'])
    def initialize(self, event):
        event.population_view.update(pd.DataFrame({
            'bmi_percentile': self.randomness.get_draw(event.index)*0.98+0.01,
            'bmi': np.full(len(event.index), 20.0)
        }))

    @listens_for('time_step__prepare', priority=8)
    @uses_columns(['bmi', 'bmi_percentile'], 'alive')
    def update_body_mass_index(self, event):
        new_bmi = self.bmi_distributions(event.index)(event.population.bmi_percentile)
        event.population_view.update(pd.Series(new_bmi, name='bmi', index=event.index))
