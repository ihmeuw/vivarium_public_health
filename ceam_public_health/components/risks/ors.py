# ~/ceam_public_health/components/ors.py

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


class Ors:
    """
    Model ORS. Simulants with diarrhea will receive ORS based on whether their `handwashing_without_soap_susceptibility` is less than the current prevalence of handwashing without soap for their demographic.

    Population Columns
    ------------------
    handwashing_without_soap_susceptibility
        Likelihood that a simulant will smoke
    """

    def setup(self, builder):

        # filter the pop so that only people with diarrhea can get ORS
        # columns = ['diarrhea']
        # self.population_view = builder.population_view(columns, 'alive')

        # self.ors_exposure = builder.value('ors_exposure')
        # self.ors_exposure.source = builder.lookup(get_exposures(238)) # USING THE HANDWASHING RISK ID FOR NOW, CHANGE TO ORS WHEN THE DATA IS MADE AVAILABLE!!!

        # USING THE HANDWASHING RISK ID FOR NOW, CHANGE TO ORS WHEN THE DATA IS MADE AVAILABLE!!!
        self.exposure = builder.lookup(get_exposures(risk_id=238))

        self.randomness = builder.randomness('ors')

        # USING THE HANDWASHING RISK ID FOR NOW, CHANGE TO ORS WHEN THE DATA IS MADE AVAILABLE!!!
        effect_function = categorical_exposure_effect(builder.lookup(get_exposures(risk_id=238)), 'ors_susceptibility')
        risk_effects = make_gbd_risk_effects(238, [
            (302, 'diarrhea_due_to_rotaviral_entiritis'),
            ], 'morbidity', effect_function)

        return risk_effects


        effect_function = categorical_exposure_effect(builder.lookup(get_exposures(risk_id=238)), 'ors_susceptibility')
        risk_effects = make_gbd_risk_effects(238, [
            (302, 'diarrhea_due_to_adenovirus'),
            ], 'morbidity', effect_function)

        return risk_effects


        effect_function = categorical_exposure_effect(builder.lookup(get_exposures(risk_id=238)), 'ors_susceptibility')
        risk_effects = make_gbd_risk_effects(238, [
            (302, 'diarrhea_due_to_norovirus'),
            ], 'morbidity', effect_function)

        return risk_effects


    @listens_for('initialize_simulants')
    @uses_columns(['ors_susceptibility'])
    def load_susceptibility(self, event):
        event.population_view.update(pd.Series(self.randomness.get_draw(event.index), name='ors_susceptibility'))

# End.
