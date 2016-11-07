import os.path
from functools import partial

import pandas as pd
import numpy as np

from ceam import config

from ceam.framework.event import listens_for
from ceam.framework.values import modifies_value
from ceam.framework.population import uses_columns

from ceam_inputs import get_pafs, get_relative_risks, get_exposures


class Unsafe_Sanitation:
    """
    Model unsafe sanitation. Simulants will be exposed to unsafe sanitation at any moment based on whether their `unsafe_sanitation_susceptibility` is less than
    the current unsafe sanitation prevalence for their demographic.

    Population Columns
    ------------------
    unsafe_sanitation_susceptibility
        Likelihood simulant is exposed to unsafe sanitation
    """

    def setup(self, builder):

        self.exposure = builder.lookup(get_exposures(risk_id=84))

        self.load_relative_risks(builder)

        builder.modifies_value(partial(self.incidence_rates, rr_lookup=self.ihd_rr), 'incidence_rate.diarrhea')

        self.load_pafs(builder)

        builder.modifies_value(partial(self.population_attributable_fraction, paf_lookup=self.ihd_paf), 'paf.diarrhea')

        self.randomness = builder.randomness('unsafe_sanitation')

    # FIXME: Think this should listen for time step as well. We want a simulant's status to be able to change each time step
    @listens_for('initialize_simulants')
    # TODO: susceptibility isn't the term we want to use here. need a better term, but can't think of what we want right now -- Everett 11/7
    @uses_columns(['unsafe_sanitation_susceptibility'])
    def load_susceptibility(self, event):
        # TODO: Confirm what the *.98 + .01 is doing in line below
        event.population_view.update(pd.Series(self.randomness.get_draw(event.index)*0.98+0.01, name='unsafe_sanitation_susceptiblity'))

    def load_relative_risks(self, builder):
        self.diarrhea_rr = builder.lookup(get_relative_risks(risk_id=84, cause_id=302))

    def load_pafs(self, builder):
        self.diarrhea_paf = builder.lookup(get_pafs(risk_id=84, cause_id=302))

    def population_attributable_fraction(self, index, paf_lookup):
        paf = paf_lookup(index)
        return paf

    @uses_columns(['unsafe_sanitation_susceptibility'])
    def incidence_rates(self, index, rates, population_view, rr_lookup):
        population = population_view.get(index)
        rr = rr_lookup(index)

        sims_exposed_to_unsafe_sanitation = population.unsafe_sanitation_susceptibility < self.exposure(index)
        rates *= rr.values**sims_exposed_to_unsafe_sanitation.values
        return rates

# End.
