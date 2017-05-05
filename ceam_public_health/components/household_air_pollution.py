# ~/ceam/ceam/modules/household_air_pollution.py

import os.path
from functools import partial

import pandas as pd
import numpy as np

from ceam import config

from ceam.framework.event import listens_for
from ceam.framework.values import modifies_value
from ceam.framework.population import uses_columns

from ceam_inputs import get_exposures, make_gbd_risk_effects

from ceam_public_health.util.risk import categorical_exposure_effect

class HouseholdAirPollution:
    """
    Model household air pollution.

    Population Columns
    ------------------
    hap_susceptibility
        Likelihood that a simulant will be exposed to HAP
    """

    def setup(self, builder):

        self.exposure = builder.lookup(get_exposures(risk_id=87))

        self.randomness = builder.randomness('household_air_pollution')

        effect_function = categorical_exposure_effect(builder.lookup(get_exposures(risk_id=87)), 'hap_susceptibility')
        risk_effects = make_gbd_risk_effects(87, [
            (493, 'heart_attack'),
            (496, 'hemorrhagic_stroke'),
            (495, 'ischemic_stroke'),
            ], effect_function)

        return risk_effects

    @listens_for('initialize_simulants')
    @uses_columns(['hap_susceptibility'])
    def load_susceptibility(self, event):
        event.population_view.update(pd.Series(self.randomness.get_draw(event.index)*0.98+0.01, name='hap_susceptibility'))

# End.
