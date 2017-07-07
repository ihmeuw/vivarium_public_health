import numpy as np
import pandas as pd

from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns
from ceam.framework.state_machine import State
from ceam.framework.values import modifies_value
from ceam_inputs import (get_etiology_specific_incidence, get_severe_diarrhea_excess_mortality,
                         get_cause_specific_mortality, get_disability_weight, get_severity_splits, causes)

from ceam_public_health.disease import RateTransition, DiseaseModel

from .data_transformations import get_duration_in_days


ETIOLOGIES = ['shigellosis',
              'cholera',
              'other_salmonella',
              'EPEC', 'ETEC',
              'campylobacter',
              'amoebiasis',
              'cryptosporidiosis',
              'rotaviral_entiritis',
              'aeromonas',
              'clostridium_difficile',
              'norovirus',
              'adenovirus',
              'unattributed_diarrhea']

event_count_columns = [eti + '_event_count' for eti in ETIOLOGIES]


class DiarrheaEtiologyState(State):
    """
    Sets up a diarrhea etiology state (e.g. a column that states that simulant
    either has diarrhea due to rotavirus or does not have diarrhea due to
    rotavirus). Child class of State.

    Parameters
    ----------
    state_id: str
        string that describes the etiology state.
    """
    def __init__(self, state_id):
        State.__init__(self, state_id)
        self.state_id = state_id
        self.event_count_column = state_id + '_event_count'

    def setup(self, builder):
        columns = [self.event_count_column]
        self.population_view = builder.population_view(columns, "alive == 'alive'")
        return super().setup(builder)

    @listens_for('initialize_simulants')
    def load_population_columns(self, event):
        self.population_view.update(pd.Series(0, index=event.index, name=self.event_count_column))

    @modifies_value('metrics')
    @uses_columns(event_count_columns)
    def metrics(self, index, metrics, population_view):
        """Output metrics counting the number of cases of diarrhea
        and number of cases overall of diarrhea due to each pathogen
        """
        metrics[self.event_count_column] = population_view.get(index)[self.event_count_column].sum()
        return metrics


# TODO: Eventually we may want to include transitions to non-fully healthy
#     states (e.g. malnourished and stunted health states)
# TODO: Eventually may want remission rates can be different across diarrhea
# due to the different etiologies
class DiarrheaBurden:
    """
    This class accomplishes several things.
        1) deletes the diarrhea csmr from the background mortality rate
        2) assigns an elevated mortality to people with severe diarrhea
        3) assigns disability weight
        4) move people into the diarrhea state
        5) determines when a simulant should remit out of the diarrhea state

    Parameters
    ----------
    excess_mortality_data: df
        df with excess mortality rate for each age, sex, year, loc
    mild_disability_weight: float
        disability weight associated with mild diarrhea
    moderate_disability_weight: float
        disability weight associated with moderate diarrhea
    severe_disability_weight: float
        disability weight associated with severe diarrhea
    mild_severity_split: float
        proportion of diarrhea cases that are mild
    moderate_severity_split: float
        proportion of diarrhea cases that are moderate
    severe_severity_split: float
        proportion of diarrhea cases that are severe
    duration_data: df
        df with duration data (in days) for each age, sex, year, loc
    """
    def __init__(self, excess_mortality_data, csmr_data,
                 mild_disability_weight, moderate_disability_weight,
                 severe_disability_weight, mild_severity_split,
                 moderate_severity_split, severe_severity_split,
                 duration_data):
        self.excess_mortality_data = excess_mortality_data
        self.csmr_data = csmr_data
        self.severity_dict = {"severe_diarrhea": severe_disability_weight,
                              "moderate_diarrhea": moderate_disability_weight,
                              "mild_diarrhea": mild_disability_weight}
        self.proportions = {"severe_diarrhea": severe_severity_split,
                            "moderate_diarrhea": moderate_severity_split,
                            "mild_diarrhea": mild_severity_split}
        self.duration_data = duration_data

    def setup(self, builder):
        self.population_view = builder.population_view(['diarrhea'], "alive == 'alive'")
        self.diarrhea_excess_mortality = builder.rate('excess_mortality.diarrhea')
        self.diarrhea_excess_mortality.source = builder.lookup(self.excess_mortality_data)
        self.duration = builder.value('duration.diarrhea')
        self.duration.source = builder.lookup(self.duration_data)
        self.randomness = builder.randomness('determine_diarrhea_severity')

    @listens_for('initialize_simulants')
    @uses_columns(['diarrhea', 'diarrhea_event_time', 'diarrhea_event_end_time', 'diarrhea_event_count'])
    def create_columns(self, event):
        event.population_view.update(pd.DataFrame({'diarrhea': pd.Series('healthy', index=event.index),
                                                   'diarrhea_event_count': pd.Series(0, index=event.index),
                                                   'diarrhea_event_time': pd.Series(pd.NaT, index=event.index),
                                                   'diarrhea_event_end_time': pd.Series(pd.NaT, index=event.index)}))

    @modifies_value('mortality_rate')
    @uses_columns(['diarrhea'], "alive == 'alive'")
    def mortality_rates(self, index, rates_df, population_view):
        population = population_view.get(index)
        rates_df['death_due_to_severe_diarrhea'] = (
            self.diarrhea_excess_mortality(population.index, skip_post_processor=True)
            * (population['diarrhea'] == 'severe_diarrhea'))
        return rates_df

    @modifies_value('csmr_data')
    def get_csmr(self):
        return self.csmr_data

    @modifies_value('disability_weight')
    def disability_weight(self, index):
        population = self.population_view.get(index)
        disability_weights = pd.Series(np.zeros(len(index), dtype=float), index=index)
        for severity in ["mild_diarrhea", "moderate_diarrhea", "severe_diarrhea"]:
            severity_index = population.query("diarrhea == '{}'".format(severity)).index
            disability_weights[severity_index] = self.severity_dict[severity]
        return disability_weights

    @listens_for('time_step', priority=6)
    @uses_columns(['diarrhea', 'diarrhea_event_time', 'diarrhea_event_count', 'age', 'sex']
                  + ETIOLOGIES + event_count_columns, 'alive == "alive" and diarrhea == "healthy"')
    def move_people_into_diarrhea_state(self, event):
        """
        Determines who should move from the healthy state to the diarrhea state
        and counts both cases of diarrhea and cases of diarrhea due to specific
        etiologies

        Assumes only healthy people can get diarrhea.
        """
        pop = event.population
        for etiology in ETIOLOGIES:
            pop.loc[pop['{}'.format(etiology)] == etiology, 'diarrhea'] = 'diarrhea'
            pop.loc[pop['{}'.format(etiology)] == etiology, '{}_event_count'.format(etiology)] += 1

        affected_pop = pop.query("diarrhea == 'diarrhea'")
        affected_pop.loc[:, 'diarrhea_event_count'] += 1
        affected_pop.loc[:, 'diarrhea_event_time'] = pd.Timestamp(event.time)
        choices, weights = zip(*self.proportions.items())
        affected_pop.loc[:, 'diarrhea'] = self.randomness.choice(affected_pop.index, choices, weights)

        event.population_view.update(affected_pop)

    @uses_columns(['diarrhea', 'diarrhea_event_time', 'diarrhea_event_end_time'] + ETIOLOGIES,
                  'alive == "alive" and diarrhea != "healthy"')
    @listens_for('time_step', priority=8)
    def apply_remission(self, event):
        affected_population = event.population

        if not affected_population.empty:
            duration_series = pd.to_timedelta(self.duration(affected_population.index), unit='D')
            affected_population['diarrhea_event_end_time'] = (duration_series
                                                              + affected_population['diarrhea_event_time'])
            current_time = pd.Timestamp(event.time)
            affected_population.loc[affected_population['diarrhea_event_end_time']
                                    <= current_time, 'diarrhea'] = 'healthy'
            for etiology in ETIOLOGIES:
                affected_population['{}'.format(etiology)] = 'healthy'

        event.population_view.update(affected_population[ETIOLOGIES + ['diarrhea', 'diarrhea_event_end_time']])


def diarrhea_factory():
    """
    Factory that moves people from an etiology state to the diarrhea state and
        uses functions above to apply excess mortality and remission
    """
    list_of_modules = []
    dict_of_etiologies_and_eti_risks = {name: causes[name].gbd_cause if name != 'unattributed_diarrhea' else 'unattributed_diarrhea'
                                        for name in ETIOLOGIES}

    for pathogen, risk_id in dict_of_etiologies_and_eti_risks.items():
        module_ = DiseaseModel(pathogen)

        healthy = State('healthy', key=pathogen)

        etiology_state = DiarrheaEtiologyState(pathogen)
        etiology_specific_incidence = get_etiology_specific_incidence(eti_risk_id=risk_id, cause_id=302, me_id=1181)
        transition = RateTransition(etiology_state, pathogen, etiology_specific_incidence)

        healthy.transition_set.append(transition)
        healthy.allow_self_transitions()
        module_.states.extend([healthy, etiology_state])

        list_of_modules.append(module_)

    diarrhea_burden = DiarrheaBurden(excess_mortality_data=get_severe_diarrhea_excess_mortality(),
                                     csmr_data=get_cause_specific_mortality(causes.diarrhea.gbd_cause),
                                     mild_disability_weight=get_disability_weight(healthstate_id=355),
                                     moderate_disability_weight=get_disability_weight(healthstate_id=356),
                                     severe_disability_weight=get_disability_weight(healthstate_id=357),
                                     mild_severity_split=get_severity_splits(1181, 2608),
                                     moderate_severity_split=get_severity_splits(1181, 2609),
                                     severe_severity_split=get_severity_splits(1181, 2610),
                                     duration_data=get_duration_in_days(1181))

    return list_of_modules + [diarrhea_burden]

