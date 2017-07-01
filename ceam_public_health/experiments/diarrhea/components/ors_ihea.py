import pandas as pd
import numpy as np

from ceam import config
from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns
from ceam.framework.values import modifies_value

from ceam_inputs import (get_diarrhea_visit_costs, get_ors_exposures,
                         get_ors_relative_risks, get_ors_pafs)

ha_given_ors_p = 0.514
ha_given_no_ors_p = 0.189


def make_ors_components():
    paf = get_ors_pafs()
    rr = get_ors_relative_risks()
    if config.ors.run_intervention:
        cost = get_ors_costs()
    elif not config.ors.run_intervention:
        cost = get_diarrhea_costs()
    exposure = get_ors_exposures()
    components = [Ors(paf, rr, exposure, cost)]
    if config.ors.run_intervention:
        exposure_increase = config.ors.ors_exposure_increase_above_baseline
        components += [IncreaseOrsExposure(exposure_increase)]
    return components


class IncreaseOrsExposure:
    def __init__(self, exposure_increase):
        self.exposure_increase = exposure_increase

    def setup(self, builder):
        pass

    @modifies_value('exposure.ors')
    def increase_exposure(self, index, exposure):
        exposure.loc[index, 'cat1'] -= self.exposure_increase
        exposure.loc[index, 'cat2'] += self.exposure_increase


class Ors:
    def __init__(self, paf_data, rr_data, exposure_data, cost_data):
        self._paf_data = paf_data
        self._rr_data = rr_data
        self._exposure_data = exposure_data
        self.cost = cost_data

    def setup(self, builder):
        self.paf = builder.value('paf.ors')
        self.paf.source = builder.lookup(self._paf_data, parameter_columns=('year',))
        self.rr = builder.value('rr.ors')
        self.rr.source = builder.lookup(self._rr_data, parameter_columns=('year',))
        self.exposure = builder.value('exposure.ors')
        self.exposure.source = builder.lookup(self._exposure_data, parameter_columns=('year',))

        self.cost = get_diarrhea_visit_costs()

        self.randomness = builder.randomness('ors_susceptibility')
        self.ha_randomness = builder.randomness('diarrhea_healthcare_access')

    @listens_for('initialize_simulants')
    @uses_columns(['ors_count', 'ors_propensity', 'ors_visit_cost', 'ors_working', 'ors_end_time'])
    def load_columns(self, event):
        length = len(event.index)
        df = pd.DataFrame({'ors_count': [0]*length,
                           'ors_propensity': self.randomness.get_draw(event.index),
                           'ors_end_time': [pd.NaT]*length,
                           'ors_working': [0]*length,
                           'ors_visit_cost': [0.0]*length}, index=event.index)
        event.population_view.update(df)

    @listens_for('time_step', priority=7)
    @uses_columns(['ors_propensity', 'diarrhea_event_time', 'ors_working', 'ors_end_time',
                   'ors_count', 'ors_visit_cost'], 'alive', 'diarrhea' == 'care_sought')
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

        bool_list = [c + '_bool' for c in categories]
        for col in categories:
            exp['{}_bool'.format(col)] = exp['{}'.format(col)] < exp['ors_propensity']
        exp['exposure_category'] = 'cat' + (exp[bool_list].sum(axis=1) + 1).astype(str)
        exp = exp[['exposure_category']]

        pop = pop.join(exp)

        recieved_ors = pop.query('exposure_category == "cat2"').index
        pop.loc[recieved_ors, 'ors_working'] = 1

        access_with_ors = self.ha_randomness.filter_for_probability(recieved_ors, ha_given_ors_p)
        access_without_ors = self.ha_randomness.filter_for_probability(pop.index.difference(recieved_ors), ha_given_no_ors_p)

        current_year = pd.Timestamp(event.time).year
        current_cost = self.cost.query("year_id == {}".format(current_year)).set_index(['year_id']).loc[current_year]['cost']

        pop.loc[access_with_ors.union(access_without_ors), 'ors_visit_cost'] += current_cost

        pop.loc[recieved_ors, 'ors_end_time'] = pop['diarrhea_event_end_time']
        pop.loc[recieved_ors, 'ors_count'] += 1

        event.population_view.update(pop)

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
            rates.loc[ors_not_working_index] *= self.rr(ors_not_working_index)[['cat1']].values

        return rates

    @modifies_value('metrics')
    @uses_columns(['ors_count', 'ors_visit_cost'])
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
            'ors_count' and 'ors_visit_cost'
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
