import pandas as pd, numpy as np

from ceam.framework.event import listens_for
from ceam.framework.values import modifies_value

from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns
from ceam.framework.values import modifies_value
from ceam import config
from ceam.framework.randomness import choice

from ceam_public_health.util import natural_key, naturally_sort_df, assign_exposure_categories, assign_relative_risk_value
from ceam_inputs import get_exposures, make_gbd_risk_effect

import os.path
from functools import partial

# 2 things that make ORS different from other risk factors we've dealt with
# 1) Only people with diarrhea can be exposed (not general population)
# 2) The RR affects diarrhea excess mortality, not incidence

# TODO: Vast majority of code below comes from ceam_public_health/util/risk.py. Make this code more flexible and keep it in one place


def ors_exposure_effect(exposure, susceptibility_column):
    """Factory that makes function which can be used as the exposure_effect
    for binary categorical risks

    Parameters
    ----------
    exposure : ceam.framework.lookup.TableView
        A lookup for exposure data
    susceptibility_column : str
        The name of the column which contains susceptibility data
    """
    @uses_columns([susceptibility_column, 'diarrhea'])
    def inner(rates, excess_mortality, population_view):
    
        pop = population_view.query("diarrhea == 'diarrhea'")

        exp = exposure(pop.index)

        exp, categories = naturally_sort_df(exp)

        # cumulatively sum over exposures
        exp = np.cumsum(exp, axis=1)

        exp = pop.join(exp)
        
        exp = assign_exposure_categories(exp, susceptibility_column, categories)

        df = exp.join(rr)

        df = assign_relative_risk_value(df, categories)

        return rates.loc[pop.index] *= (df.relative_risk_value.values)

    return inner


class RiskEffect:
    """RiskEffect objects bundle all the effects that a given risk has on a
    cause.
    """
    def __init__(self, rr_data, paf_data, cause, exposure_effect):
        """
        Parameters
        ----------
        rr_data : pandas.DataFrame
            A dataframe of relative risk data with age, sex, year, and rr columns
        paf_data : pandas.DataFrame
            A dataframe of population attributable fraction data with age, sex, year, and paf columns
        cause : str
            The name of the cause to effect as used in named variables like 'incidence_rate.<cause>'
        exposure_effect : callable
            A function which takes a series of incidence rates and a series of
            relative risks and returns rates modified as appropriate for this risk
        """
        self.rr_data = rr_data
        self.paf_data = paf_data
        self.cause_name = cause
        self.exposure_effect = exposure_effect

    def setup(self, builder):
        self.rr_lookup = builder.lookup(self.rr_data)
        builder.modifies_value(self.excess_mortality_rates, 'excess_mortality.diarrhea')
        builder.modifies_value(builder.lookup(self.paf_data), 'paf.{}'.format(self.cause_name))

        return [self.exposure_effect]

    def excess_mortality_rates(self, index, rates):
        rr = self.rr_lookup(index)

        return self.exposure_effect(rates, rr)
    

class ORS():
    def __init__(self, active=True):
        self.active = active

    def setup(self, builder):

        # filter the pop so that only people with diarrhea can get ORS
        # columns = ['diarrhea']
        # self.population_view = builder.population_view(columns, 'alive')

        # self.ors_exposure = builder.value('ors_exposure')
        # self.ors_exposure.source = builder.lookup(get_exposures(238)) # USING THE HANDWASHING RISK ID FOR NOW, CHANGE TO ORS WHEN THE DATA IS MADE AVAILABLE!!!

        # USING THE HANDWASHING RISK ID FOR NOW, CHANGE TO ORS WHEN THE DATA IS MADE AVAILABLE!!!
        self.exposure = builder.lookup(get_exposures(risk_id=238))

        if self.active:
            # add exposure above baseline increase in intervention scenario
            ors_exposure_increase_above_baseline = config.getfloat('ORS', 'ors_exposure_increase_above_baseline')
            self.exposure['cat1'] += ors_exposure_increase_above_baseline
            self.exposure['cat2'] -= ors_exposure_increase_above_baseline

        # define the ors exposure value, which will be manipulated in the intervention
        self.ors_exposure = builder.value('ors_exposure')

        self.randomness = builder.randomness('ors')

        # USING THE HANDWASHING RISK ID FOR NOW, CHANGE TO ORS WHEN THE DATA IS MADE AVAILABLE!!!
        effect_function = ors_exposure_effect(builder.lookup(get_exposures(risk_id=238)), 'ors_susceptibility')
        risk_effects_rota = make_gbd_risk_effects(238, [
            (302, 'diarrhea_due_to_rotaviral_entiritis'),
            ], 'morbidity', effect_function)

        effect_function = ors_exposure_effect(builder.lookup(get_exposures(risk_id=238)), 'ors_susceptibility')
        risk_effects_adeno = make_gbd_risk_effects(238, [
            (302, 'diarrhea_due_to_adenovirus'),
            ], 'morbidity', effect_function)

        effect_function = ors_exposure_effect(builder.lookup(get_exposures(risk_id=238)), 'ors_susceptibility')
        risk_effects_noro = make_gbd_risk_effects(238, [
            (302, 'diarrhea_due_to_norovirus'),
            ], 'morbidity', effect_function)

        return risk_effects_rota, risk_effects_adeno, risk_effects_noro
