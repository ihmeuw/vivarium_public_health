import pandas as pd, numpy as np

from ceam.framework.event import listens_for
from ceam.framework.values import modifies_value

from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns
from ceam.framework.values import modifies_value
from ceam import config
from ceam.framework.randomness import choice

from ceam_public_health.util.risk import natural_key, naturally_sort_df, assign_exposure_categories, assign_relative_risk_value
from ceam_inputs import get_exposures, get_relative_risks, get_pafs

import os.path
from functools import partial
import pdb

# FIXME: Won't need import below once ORS exposure and RR estimates are uploaded to the database
from ceam_inputs.gbd_ms_auxiliary_functions import expand_ages_for_dfs_w_all_age_estimates, normalize_for_simulation

# 2 things that make ORS different from other risk factors we've dealt with
# 1) Only people with diarrhea can be exposed (not general population)
# 2) The RR/PAF affects diarrhea excess mortality, not incidence

# TODO: Vast majority of code below comes from ceam_public_health/util/risk.py. Make this code more flexible and keep it in one place

# TODO: Incorporate PAFs


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
    @uses_columns([susceptibility_column, 'diarrhea', 'ors_unit_cost', 'ors_cost_to_administer', 'ors_count'])
    def inner(rates, rr, population_view):
   
        population = population_view.get(rr.index)
 
        pop = population.query("diarrhea == 'diarrhea'").copy()

        exp = exposure(pop.index)

        exp, categories = naturally_sort_df(exp)

        # cumulatively sum over exposures
        exp = np.cumsum(exp, axis=1)

        exp = pop.join(exp)
        
        exp = assign_exposure_categories(exp, susceptibility_column, categories)

        df = exp.join(rr)

        df = assign_relative_risk_value(df, categories)

        # costs and counts
        received_ors_index = df.query("exposure_category == 'cat1'").index

        #FIXME: We don't want to use a placeholder after the ORS data is ready
        df.loc[received_ors_index, 'relative_risk_value'] = 1 - config.getfloat('ORS', 'ors_effectiveness')

        # TODO: Make sure the categories make sense. Exposure to ORS should decrease risk (i.e. RR should be less than 1)
        rates.loc[pop.index] *= (df.relative_risk_value.values)

        if not pop.loc[received_ors_index].empty:
            pop.loc[received_ors_index, 'ors_unit_cost'] += config.getint('ORS', 'ORS_unit_cost')
            pop.loc[received_ors_index, 'ors_cost_to_administer'] += config.getint('ORS', 'cost_to_administer_ORS')
            pop.loc[received_ors_index, 'ors_count'] += 1

            population_view.update(pop)

        return rates

    return inner


class ORSRiskEffect:
    """RiskEffect objects bundle all the effects that a given risk has on a
    cause.
    """
    def __init__(self, rr_data, exposure_effect):
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
        self.exposure_effect = exposure_effect

    def setup(self, builder):
        self.rr_lookup = builder.lookup(self.rr_data)
        builder.modifies_value(self.excess_mortality_rates, 'excess_mortality.diarrhea')

        return [self.exposure_effect]

    # TODO: This was defined as incidence rates, but it just grabs the relative risk, correct? How do the two return statements factor in?
    # TODO: Incorporate PAFs below
    def excess_mortality_rates(self, index, rates):
        rr = self.rr_lookup(index)

        return self.exposure_effect(rates, rr)


def make_gbd_risk_effects(risk_id, causes, rr_type, effect_function):
    return [ORSRiskEffect(
        get_relative_risks(risk_id=risk_id, cause_id=cause_id, rr_type=rr_type),
        effect_function)
        for cause_id, cause_name in causes]
   

# FIXME: Won't need function below once ORS exposure and RR estimates are uploaded to the database
def get_ors_exposure():
    covariate_estimates_input = pd.read_csv("/share/epi/risk/bmgf/draws/exp/diarrhea_ors.csv")

    covariate_estimates = covariate_estimates_input.query("location_id == {}".format(config.getint('simulation_parameters', 'location_id'))).copy()

    expanded = expand_ages_for_dfs_w_all_age_estimates(covariate_estimates)

    expanded_estimates = expanded.query("year_id >= {ys} and year_id <= {ye}".format(ys = config.getint('simulation_parameters', 'year_start'), ye = config.getint('simulation_parameters', 'year_end'))).copy()

    keepcols = ['year_id', 'sex_id', 'age', 'cat1', 'cat2']

    expanded_estimates.rename(columns={'draw_{}'.format(config.getint('run_configuration', 'draw_number')): 'cat1'}, inplace=True)

    expanded_estimates['cat2'] = 1 - expanded_estimates['cat1']

    expanded_estimates = expanded_estimates[keepcols]

    return normalize_for_simulation(expanded_estimates)
   

class ORS():
    def __init__(self, active=True):
        self.active = active


    def setup(self, builder):

        ors_exposure = get_ors_exposure()


        if self.active:
            # add exposure above baseline increase in intervention scenario
            ors_exposure_increase_above_baseline = config.getfloat('ORS', 'ors_exposure_increase_above_baseline')
            ors_exposure['cat1'] += ors_exposure_increase_above_baseline
            ors_exposure['cat2'] -= ors_exposure_increase_above_baseline

        self.exposure = builder.lookup(ors_exposure)

        self.randomness = builder.randomness('ors')

        # FIXME: I'm using the handwashing rei_id right now -- 238 -- for RR but I'm manually overwriting the RR values to numbers that make sense for ORS. Once we have the ORS rei_id, I can update
        effect_function = ors_exposure_effect(self.exposure, 'ors_susceptibility')
        risk_effects = make_gbd_risk_effects(238, [
            (302, 'diarrhea'),
            ], 'mortality', effect_function)

        return risk_effects


    # TODO: May want to rethink susceptibility column getting assigned at birth. Distribution of susceptibility may differ for people that actually get diarrhea
    @listens_for('initialize_simulants')
    @uses_columns(['ors_susceptibility', 'ors_unit_cost', 'ors_cost_to_administer', 'ors_count'])
    def load_susceptibility(self, event):
        event.population_view.update(pd.Series(self.randomness.get_draw(event.index), name='ors_susceptibility'))
        event.population_view.update(pd.DataFrame({'ors_unit_cost': np.zeros(len(event.index), dtype=int)}))
        event.population_view.update(pd.DataFrame({'ors_cost_to_administer': np.zeros(len(event.index), dtype=int)}))
        event.population_view.update(pd.DataFrame({'ors_count': np.zeros(len(event.index), dtype=int)}))


    @modifies_value('metrics')
    @uses_columns(['ors_count', 'ors_unit_cost', 'ors_cost_to_administer'])
    def metrics(self, index, metrics, population_view):
        population = population_view.get(index)

        metrics['ors_unit_cost'] = population['ors_unit_cost'].sum()
        metrics['number_of_days_ors_is_supplied'] = population['ors_count'].sum()
        metrics['ors_cost_to_administer'] = population['ors_cost_to_administer'].sum()

        return metrics


# End.
