# ~/ceam_public_health/components/underweight.py

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

class Underweight:
    """
    Model underweight. Simulants will be underweight at any moment based on whether their `underweight_susceptibility` is less than the current prevalence of underweight for their demographic.

    Population Columns
    ------------------
    underweight_susceptibility
        Likelihood that a simulant will be underweight
    """

    def setup(self, builder):

        self.randomness = builder.randomness('underweight')

        list_of_etiologies = ['diarrhea_due_to_shigellosis', 'diarrhea_due_to_cholera', 'diarrhea_due_to_other_salmonella', 'diarrhea_due_to_EPEC', 'diarrhea_due_to_ETEC', 'diarrhea_due_to_campylobacter', 'diarrhea_due_to_amoebiasis', 'diarrhea_due_to_cryptosporidiosis', 'diarrhea_due_to_rotaviral_entiritis', 'diarrhea_due_to_aeromonas', 'diarrhea_due_to_clostridium_difficile', 'diarrhea_due_to_norovirus', 'diarrhea_due_to_adenovirus']

        list_of_tuples = [(302, i) for i in list_of_etiologies]

        effect_function = categorical_exposure_effect(builder.lookup(get_exposures(risk_id=94)), 'underweight_susceptibility')
        risk_effects = make_gbd_risk_effects(94, list_of_tuples, 'morbidity', effect_function)

        return risk_effects

    @listens_for('initialize_simulants')
    @uses_columns(['underweight_susceptibility'])
    def load_susceptibility(self, event):
        event.population_view.update(pd.Series(self.randomness.get_draw(event.index), name='underweight_susceptibility'))

# End.
