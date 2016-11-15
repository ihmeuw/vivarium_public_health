from functools import partial

import pandas as pd
import numpy as np

from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns

from ceam_inputs import get_bmi_distributions, get_relative_risks, get_pafs

class BodyMassIndex:
    """Model BMI

    Population Columns
    ------------------
    bmi_percentile
        Position of the simulant in the population's BMI distribution
    """

    def setup(self, builder):
        self.bmi_distributions = builder.lookup(get_bmi_distributions())
        self.randomness = builder.randomness('bmi')

        self.load_reletive_risks(builder)

        builder.modifies_value(partial(self.incidence_rates, rr_lookup=self.ihd_rr), 'incidence_rate.heart_attack')
        builder.modifies_value(partial(self.incidence_rates, rr_lookup=self.hemorrhagic_stroke_rr), 'incidence_rate.hemorrhagic_stroke')
        builder.modifies_value(partial(self.incidence_rates, rr_lookup=self.ischemic_stroke_rr), 'incidence_rate.ischemic_stroke')

        self.load_pafs(builder)

        builder.modifies_value(partial(self.population_attributable_fraction, paf_lookup=self.ihd_paf), 'paf.heart_attack')
        builder.modifies_value(partial(self.population_attributable_fraction, paf_lookup=self.hemorrhagic_stroke_paf), 'paf.hemorrhagic_stroke')
        builder.modifies_value(partial(self.population_attributable_fraction, paf_lookup=self.ischemic_stroke_paf), 'paf.ischemic_stroke')



    @listens_for('initialize_simulants')
    @uses_columns(['bmi_percentile', 'bmi'])
    def initialize(self, event):
        event.population_view.update(pd.DataFrame({
            'bmi_percentile': self.randomness.get_draw(event.index)*0.98+0.01,
            'bmi': np.full(len(event.index), 20)
        }))

    def load_reletive_risks(self, builder):
        self.ihd_rr = builder.lookup(get_relative_risks(risk_id=108, cause_id=493))
        self.hemorrhagic_stroke_rr = builder.lookup(get_relative_risks(risk_id=108, cause_id=496))
        self.ischemic_stroke_rr = builder.lookup(get_relative_risks(risk_id=108, cause_id=495))

    def load_pafs(self, builder):
        self.ihd_paf = builder.lookup(get_pafs(risk_id=108, cause_id=493))
        self.hemorrhagic_stroke_paf = builder.lookup(get_pafs(risk_id=108, cause_id=496))
        self.ischemic_stroke_paf = builder.lookup(get_pafs(risk_id=108, cause_id=495))

    def population_attributable_fraction(self, index, paf_lookup):
        paf = paf_lookup(index)
        return paf

    @listens_for('time_step__prepare', priority=8)
    @uses_columns(['bmi', 'bmi_percentile'], 'alive')
    def update_body_mass_index(self, event):
        new_bmi = self.bmi_distributions(event.index)(event.population.bmi_percentile)
        event.population_view.update(pd.Series(new_bmi, name='bmi', index=event.index))

    @uses_columns(['bmi'])
    def incidence_rates(self, index, rates, population_view, rr_lookup):
        population = population_view.get(index)
        rr = rr_lookup(index)

        rates *= np.maximum(rr.values**((population.bmi - 21) / 5).values, 1)
        return rates
