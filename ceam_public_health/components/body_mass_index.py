import pandas as pd
import numpy as np

from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns

from ceam_inputs import get_bmi_distributions, risk_factors

from ceam_public_health.util.risk import continuous_exposure_effect, make_risk_effects


class BodyMassIndex:
    """Model BMI

    Population Columns
    ------------------
    bmi_percentile
        Position of the simulant in the population's BMI distribution
    """

    def setup(self, builder):
        self.default = 20
        self.name = 'body_mass_index'
        self.risk = risk_factors[self.name]
        self.bmi_distributions = builder.lookup(get_bmi_distributions())
        self.randomness = builder.randomness(self.name)

        effect_function = continuous_exposure_effect(self.name, tmrl=self.risk.tmrl, scale=self.risk.scale)
        risk_effects = make_risk_effects(self.risk.gbd_risk,
                                         [(c.gbd_cause, c) for c in self.risk.effected_causes],
                                         effect_function,
                                         self.name)
        return risk_effects

    @listens_for('initialize_simulants')
    @uses_columns(['body_mass_index_percentile', 'body_mass_index'])
    def initialize(self, event):
        event.population_view.update(pd.DataFrame({
            '{}_percentile'.format(self.name): self.randomness.get_draw(event.index)*0.98+0.01,
            self.name: np.full(len(event.index), self.default)
        }))

    @listens_for('time_step__prepare', priority=8)
    @uses_columns(['body_mass_index_percentile', 'body_mass_index'], 'alive')
    def update_body_mass_index(self, event):
        new_bmi = self.bmi_distributions(event.index)(event.population.bmi_percentile)
        event.population_view.update(pd.Series(new_bmi, name=self.name, index=event.index))
