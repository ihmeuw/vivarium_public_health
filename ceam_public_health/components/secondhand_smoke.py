# ~/ceam/ceam/modules/secondhand_smoke.py

import pandas as pd

from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns

from ceam_inputs import get_exposures

from ceam_public_health.util.risk import categorical_exposure_effect, make_risk_effects

class SecondhandSmoke:
    """
    Models secondhand smoke.

    Population Columns
    ------------------
    shs_susceptibility
        Likelihood that a simulant will smoke
    """

    def setup(self, builder):

        self.exposure = builder.lookup(get_exposures(risk_id=100))

        self.randomness = builder.randomness('secondhand_smoke')

        effect_function = categorical_exposure_effect(builder.lookup(get_exposures(risk_id=100)), 'shs_susceptibility')
        risk_effects = make_risk_effects(100, [
            (493, 'heart_attack'),
            (496, 'hemorrhagic_stroke'),
            (495, 'ischemic_stroke'),
            ], effect_function, 'secondhand_smoke')

        return risk_effects

    @listens_for('initialize_simulants')
    @uses_columns(['shs_susceptibility'])
    def load_susceptibility(self, event):
        event.population_view.update(pd.Series(self.randomness.get_draw(event.index)*0.98+0.01, name='shs_susceptibility'))

# End.
