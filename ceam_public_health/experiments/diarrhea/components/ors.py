import pandas as pd
import numpy as np

from ceam import config
from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns
from ceam.framework.values import modifies_value

from ceam_inputs import (get_diarrhea_visit_costs, get_ors_exposures,
                         get_ors_relative_risks, get_ors_pafs)


class Ors:
    """
    The Ors class accomplishes several things
    1) Reads in all ors risk data (pafs, relative risks, and exposures) and
        outpatient visit costs (we are setting the unit cost of ors to be the
        cost of an outpatient visit)
        #FIXME: Should was also include unit cost estimate for ors? Certainly
        some locations wouldn't have the ors cost baked into the visit cost
    2) If config.ors.run_intervention is set to True, the exposure will be
        updated based on the value in
        config.ors.ors_exposure_increase_above_baseline
    3) Creates the columns necessary to the component
    4) Determines which simulants are currently receiving ors
    5) Sets the lack of ors-deleted excess mortality rate for all simulants. For
        simulants that do not receive ors, we multiply the lack of ors-deleted
        mortality rate by the relative risk
    6) Outputs metrics for ors costs and counts
    """
    def setup(self, builder):
        self.paf = builder.value('ors_population_attributable_fraction')
        self.paf.source = builder.lookup(get_ors_pafs())
        self.rr = builder.value('ors_relative_risk')
        self.rr.source = builder.lookup(get_ors_relative_risks())
        self.cost = get_diarrhea_visit_costs()

        ors_exposure = get_ors_exposures()
        if config.ors.run_intervention:
            exposure_increase = config.ors.ors_exposure_increase_above_baseline
            ors_exposure['cat1'] -= exposure_increase
            ors_exposure['cat2'] += exposure_increase
        self.exposure = builder.value('exposure.ors')
        self.exposure.source = builder.lookup(ors_exposure)
        self.randomness = builder.randomness('ors_susceptibility')
        self.ha_randomness = builder.randomness('diarrhea_healthcare_access')

        self.cost = get_diarrhea_visit_costs()

    @listens_for('initialize_simulants')
    @uses_columns(['ors_count', 'ors_propensity', 'ors_visit_cost',
                   'ors_working', 'ors_end_time'])
    def load_columns(self, event):
        length = len(event.index)
        df = pd.DataFrame({'ors_count': [0]*length,
                           'ors_propensity': self.randomness.get_draw(event.index),
                           'ors_end_time': [pd.NaT]*length,
                           'ors_working': [0]*length,
                           'ors_visit_cost': [0.0]*length}, index=event.index)
        event.population_view.update(df)

    @listens_for('time_step', priority=7)
    @uses_columns(['ors_propensity', 'diarrhea_event_time',
                   'diarrhea_event_end_time', 'ors_working', 'ors_end_time',
                   'ors_count', 'ors_visit_cost'], 'alive')
    def determine_who_gets_ors(self, event):
        """
        This method determines who should be seeing the benefit of ors
        """
        pop = event.population
        pop.loc[pop['ors_end_time'] <= event.time, 'ors_working'] = 0
        pop = pop.loc[pop['diarrhea_event_time'].notnull()]
        pop = pop.loc[pop['diarrhea_event_time'] == pd.Timestamp(event.time)]

        exp = self.exposure(pop.index)
        categories = sorted([c for c in exp.columns if 'cat' in c], key=lambda c: int(c.split('cat')[1]))
        exp = exp[categories]
        exp = np.cumsum(exp, axis=1)
        exp = pop.join(exp)
        exp = assign_exposure_categories(exp, 'ors_propensity', categories)
        pop = pop.join(exp)

        recieved_ors = pop.query('exposure_category == "cat2"').index
        pop.loc[recieved_ors, 'ors_working'] = 1

        ha_given_ors_p = 0.514
        ha_given_no_ors_p = 0.189

        access_with_ors = self.ha_randomness.filter_for_probability(recieved_ors, ha_given_ors_p)
        access_without_ors = self.ha_randomness.filter_for_probability(pop.index.difference(recieved_ors), ha_given_no_ors_p)

        current_year = pd.Timestamp(event.time).year
        current_cost = self.cost.query(
            "year_id == {}".format(current_year)).set_index(['year_id']).loc[current_year]['cost']

        pop.loc[access_with_ors.union(access_without_ors), 'ors_visit_cost'] += current_cost

        pop.loc[recieved_ors, 'ors_end_time'] = pop['diarrhea_event_end_time']
        pop.loc[recieved_ors, 'ors_count'] += 1


        event.population_view.update(pop)

    # FIXME: Need to ensure the mortality rates calculation happens after determine_who_gets_ors
    @modifies_value('excess_mortality.diarrhea')
    @uses_columns(['ors_working'])
    def mortality_rates(self, index, rates, population_view):
        """
        Set the lack of ors-deleted mortality rate for all simulants. For those
        exposed to the risk (the risk is the ABSENCE of ors), multiply the
        lack of ors-deleted excess mortality rate by the relative risk
        """
        pop = population_view.get(index)
        rates *= 1 - self.paf(index)
        ors_not_working_index = pop.query("ors_working == 0").index

        if not ors_not_working_index.empty:
            rates.loc[ors_not_working_index] *= self.rr(ors_not_working_index)[['cat1']].values.flatten()

        return rates

    @modifies_value('metrics')
    @uses_columns(['ors_count', 'ors_visit_cost', 'ors_facility_cost'])
    def metrics(self, index, metrics, population_view):
        """
        Update the output metrics with information regarding the vaccine
        intervention

        Parameters
        ----------
        index: pandas Index
            Index of all simulants, alive or dead

        metrics: pd.Dictionary
            Dictionary of metrics that will be printed out at the end of the
            simulation

        population_view: pd.DataFrame
            df of all simulants, alive or dead with columns
            'ors_count', 'ors_visit_cost', and 'ors_facility_cost'
        """
        population = population_view.get(index)

        metrics['ors_visit_cost'] = population['ors_visit_cost'].sum()
        metrics['ors_count'] = population['ors_count'].sum()

        return metrics


def assign_exposure_categories(df, susceptibility_column, categories):
    """Creates an 'exposure_category' column that assigns 
    simulant's exposure based on their susceptibility draw

    Parameters
    ----------
    df : pd.DataFrame
    susceptibility_column : str
    categories : list
        list of all of the category columns in df 
    """
    bool_list = [c + '_bool' for c in categories]
    for col in categories:
        df['{}_bool'.format(col)] = df['{}'.format(col)] < df[susceptibility_column]
    df['exposure_category'] = 'cat' + (df[bool_list].sum(axis=1) + 1).astype(str)

    return df[['exposure_category']]
