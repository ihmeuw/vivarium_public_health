import pandas as pd

from ceam import config
from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns
from ceam.framework.values import modifies_value

from ceam_inputs import get_ors_relative_risks, get_ors_pafs

ors_coverage = .58
ors_unit_cost = 0.50


def make_ors_components():
    paf = get_ors_pafs()  # Dataframe
    rr = get_ors_relative_risks()  # Scalar
    exposure = 1 - ors_coverage  # Scalar

    components = [Ors(paf, rr, exposure, ors_unit_cost)]

    if config.ors.run_intervention:
        components += [IncreaseOrsCoverage(config.ors.additional_coverage)]

    return components


class IncreaseOrsCoverage:
    def __init__(self, coverage_increase):
        self.coverage_increase = coverage_increase

    @modifies_value('exposure.ors')
    def increase_exposure(self, _, exposure):
        return exposure - self.coverage_increase


class Ors:
    def __init__(self, paf_data, rr_data, exposure_data, unit_cost):
        self._paf_data = paf_data
        self.rr = rr_data
        self.exposure = exposure_data
        self.unit_cost = unit_cost

    def setup(self, builder):
        self.paf = builder.value('paf.ors')
        self.paf.source = builder.lookup(self._paf_data)
        self.population_view = builder.population_view(['ors_count'])

        self.randomness = builder.randomness('ors')

    @listens_for('initialize_simulants')
    @uses_columns(['ors_count', 'ors_propensity', 'receiving_ors'])
    def load_columns(self, event):
        length = len(event.index)
        df = pd.DataFrame({'ors_count': [0]*length,
                           'ors_propensity': self.randomness.get_draw(event.index),
                           'receiving_ors': [False]*length})
        event.population_view.update(df)

    @listens_for('time_step', priority=7)
    @uses_columns(['ors_propensity', 'diarrhea', 'ors_count', 'receiving_ors'], "alive == 'alive'")
    def administer_ors(self, event):
        pop = event.population.copy()
        receiving_ors = (pop['diarrhea'] == 'care_sought') & (pop['ors_propensity'] > self.exposure)

        pop.loc[receiving_ors, 'receiving_ors'] = True
        pop.loc[~receiving_ors, 'receiving_ors'] = False
        pop.loc[receiving_ors, 'ors_count'] += 1

        event.population_view.update(pop)

    @modifies_value('excess_mortality.diarrhea')
    @uses_columns(['receiving_ors'])
    def mortality_rates(self, index, rates, population_view):
        """
        Set the lack of ors-deleted mortality rate for all simulants. For those
        exposed to the risk (the risk is the ABSENCE of ors), multiply the
        lack of ors-deleted excess mortality rate by the relative risk
        """
        # Reduce everone's excess mortality
        rates *= 1 - self.paf(index)
        # Increase the excess mortality for anyone not receiving ors.
        no_ors = ~population_view.get(index)['receiving_ors']
        rates.loc[no_ors] *= self.rr
        return rates

    @modifies_value('metrics')
    def metrics(self, index, metrics):
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

        """
        population = self.population_view.get(index)

        metrics['ors_count'] = population['ors_count'].sum()
        metrics['ors_visit_cost'] = population['ors_count'].sum() * self.unit_cost

        return metrics
