# ~/ceam_public_health/components/handwashing_without_soap.py

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

class Handwashing:
    """
    Model handwashing without soap. Simulants will be smoking at any moment based on whether their `handwashing_without_soap_susceptibility` is less than the current prevalence of handwashing without soap for their demographic.

    Population Columns
    ------------------
    handwashing_without_soap_susceptibility
        Likelihood that a simulant will smoke
    """

    def setup(self, builder):

        self.exposure = builder.lookup(get_exposures(risk_id=238))

        self.randomness = builder.randomness('handwashing_without_soap')

        effect_function = categorical_exposure_effect(builder.lookup(get_exposures(risk_id=238)), 'handwashing_without_soap_susceptibility')
        risk_effects = make_gbd_risk_effects(238, [
            # TODO: Make this not dependent on GBD! i.e. get rid of the risk id and cause id
            (302, 'severe_diarrhea_due_to_rotavirus'),
            ], effect_function)

        return risk_effects

        effect_function = categorical_exposure_effect(builder.lookup(get_exposures(risk_id=238)), 'handwashing_without_soap_susceptibility')
        risk_effects = make_gbd_risk_effects(238, [
            (302, 'severe_diarrhea_due_to_rotavirus'),
            ], effect_function)

        return risk_effects

    @listens_for('initialize_simulants')
    @uses_columns(['handwashing_without_soap_susceptibility'])
    def load_susceptibility(self, event):
        event.population_view.update(pd.Series(self.randomness.get_draw(event.index), name='handwashing_without_soap_susceptibility'))

# End.
