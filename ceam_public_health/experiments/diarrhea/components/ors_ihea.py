import pandas as pd
import numpy as np

from ceam import config
from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns
from ceam.framework.values import modifies_value

from ceam_inputs import get_ors_exposures, get_ors_relative_risks, get_ors_pafs

# Should be un-used.
ha_given_ors_p = 0.514
ha_given_no_ors_p = 0.189

ors_prevalence = 0.58
ors_exposure = 1 - ors_prevalence
ors_unit_cost = 0.50


def make_ors_components():
    paf = get_ors_pafs()
    rr = get_ors_relative_risks()
    exposure = ors_exposure

    components = [Ors(paf, rr, exposure, ors_unit_cost)]

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
    def __init__(self, paf_data, rr_data, exposure_data, unit_cost):
        self._paf_data = paf_data
        self._rr_data = rr_data
        self._exposure_data = exposure_data
        self.unit_cost = unit_cost

    def setup(self, builder):
        self.paf = builder.value('paf.ors')
        self.paf.source = builder.lookup(self._paf_data, parameter_columns=('year',))
        self.rr = builder.value('rr.ors')
        self.rr.source = builder.lookup(self._rr_data, parameter_columns=('year',))
        self.exposure = builder.value('exposure.ors')
        self.exposure.source = builder.lookup(self._exposure_data)

        self.randomness = builder.randomness('ors')

    @listens_for('initialize_simulants')
    @uses_columns(['ors_count', 'ors_propensity'])
    def load_columns(self, event):
        length = len(event.index)
        df = pd.DataFrame({'ors_count': [0]*length,
                           'ors_propensity': self.randomness.get_draw(event.index),
                           'receiving_ors': [False]*length})
        event.population_view.update(df)


    @listens_for('time_step', priority=7)
    @uses_columns(['ors_propensity', 'diarrhea', 'ors_count'], "alive == 'alive'")
    def determine_who_gets_ors(self, event):
        """
        This method determines who should be seeing the benefit of ors
        """
        healthy = event.population['diarrhea']
        pop.loc[pop['ors_end_time'] <= event.time, 'ors_working'] = 0
        pop = pop.loc[pop['diarrhea_event_time'] == pd.Timestamp(event.time)]

        exp = self.exposure(pop.index)
        categories = sorted([c for c in exp.columns if 'cat' in c])
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
