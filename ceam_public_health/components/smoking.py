# ~/ceam/ceam/modules/smoking.py

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

class Smoking:
    """
    Model smoking. Simulants will be smoking at any moment based on whether their `smoking_susceptibility` is less than
    the current smoking prevalence for their demographic.

    NOTE: This does not track whether a simulant has a history of smoking, only what their current state is.

    Population Columns
    ------------------
    smoking_susceptibility
        Likelihood that a simulant will smoke
    """

    def setup(self, builder):

        self.exposure = builder.lookup(get_exposures(risk_id=166))

        self.randomness = builder.randomness('smoking')

        effect_function = categorical_exposure_effect(builder.lookup(get_exposures(risk_id=166)), 'smoking_susceptibility')
        risk_effects = make_gbd_risk_effects(166, [
            (493, 'heart_attack'),
            (496, 'hemorrhagic_stroke'),
            (495, 'ischemic_stroke'),
            ], effect_function)

        return risk_effects


    @listens_for('initialize_simulants')
    @uses_columns(['smoking_susceptibility'])
    def load_susceptibility(self, event):
        event.population_view.update(pd.Series(self.randomness.get_draw(event.index), name='smoking_susceptibility'))

# End.
