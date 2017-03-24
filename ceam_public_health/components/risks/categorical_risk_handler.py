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

class CategoricalRiskHandler:
    """
    Model unsafe sanitation. Simulants will be in a specific exposure category based on their `unsafe_sanitation_susceptibility`.

    Population Columns
    ------------------
    unsafe_sanitation_susceptibility        
    """
    def __init__(risk_id, risk_name):
        self.risk_id = risk_id
        self.risk_name = risk_name

    def setup(self, builder):

        self.exposure = builder.value('{}.exposure'.format(self.risk_name))

        self.exposure.source = builder.lookup(get_exposures(risk_id=self.risk_id))

        self.randomness = builder.randomness(self.risk_name)

        list_of_etiologies = ['diarrhea_due_to_shigellosis', 'diarrhea_due_to_cholera', 'diarrhea_due_to_other_salmonella', 'diarrhea_due_to_EPEC', 'diarrhea_due_to_ETEC', 'diarrhea_due_to_campylobacter', 'diarrhea_due_to_amoebiasis', 'diarrhea_due_to_cryptosporidiosis', 'diarrhea_due_to_rotaviral_entiritis', 'diarrhea_due_to_aeromonas', 'diarrhea_due_to_clostridium_difficile', 'diarrhea_due_to_norovirus', 'diarrhea_due_to_adenovirus']

        list_of_tuples = [(302, i) for i in list_of_etiologies]

        effect_function = categorical_exposure_effect(self.exposure, '{}_susceptibility'.format(self.risk_name))
        risk_effects = make_gbd_risk_effects(self.risk_id, list_of_tuples, effect_function)

        return risk_effects

    def load_susceptibility(self):
        @listens_for('initialize_simulants')
        @uses_columns(['{}_susceptibility'.format(self.risk_name)])
        def load_susceptibility(self, event):
        event.population_view.update(pd.Series(self.randomness.get_draw(event.index), name='{}_susceptibility'.format(self.risk_name)))

# End.

