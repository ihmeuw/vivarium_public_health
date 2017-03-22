# ~/ceam_public_health/components/risks/unsafe_sanitation.py

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

class UnsafeSanitation:
    """
    Model unsafe sanitation. Simulants will be in a specific exposure category based on their `unsafe_sanitation_susceptibility`.

    Population Columns
    ------------------
    unsafe_sanitation_susceptibility        
    """

    def setup(self, builder):

        self.exposure = builder.value('unsafe_sanitation.exposure')

        self.exposure.source = builder.lookup(get_exposures(risk_id=84))

        self.randomness = builder.randomness('unsafe_sanitation')

        list_of_etiologies = ['diarrhea_due_to_shigellosis', 'diarrhea_due_to_cholera', 'diarrhea_due_to_other_salmonella', 'diarrhea_due_to_EPEC', 'diarrhea_due_to_ETEC', 'diarrhea_due_to_campylobacter', 'diarrhea_due_to_amoebiasis', 'diarrhea_due_to_cryptosporidiosis', 'diarrhea_due_to_rotaviral_entiritis', 'diarrhea_due_to_aeromonas', 'diarrhea_due_to_clostridium_difficile', 'diarrhea_due_to_norovirus', 'diarrhea_due_to_adenovirus']

        list_of_tuples = [(302, i) for i in list_of_etiologies]

        effect_function = categorical_exposure_effect(builder.lookup(get_exposures(risk_id=84)), 'unsafe_sanitation_susceptibility')
        risk_effects = make_gbd_risk_effects(84, list_of_tuples, effect_function)

        return risk_effects

    @listens_for('initialize_simulants')
    @uses_columns(['unsafe_sanitation_susceptibility'])
    def load_susceptibility(self, event):
        event.population_view.update(pd.Series(self.randomness.get_draw(event.index), name='unsafe_sanitation_susceptibility'))

# End.

