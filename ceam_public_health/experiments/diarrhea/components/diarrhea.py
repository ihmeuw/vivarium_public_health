import numpy as np
import pandas as pd

from ceam import config
from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns
from ceam.framework.state_machine import State
from ceam.framework.values import modifies_value
from ceam_inputs import (get_etiology_specific_incidence, get_severe_diarrhea_excess_mortality,
                         get_cause_specific_mortality, get_disability_weight, get_severity_splits)

from ceam_public_health.disease import RateTransition, DiseaseModel
from ceam_public_health.experiments.diarrhea.components.diarrhea2 import get_duration_in_days
from ceam_public_health.util import make_cols_demographically_specific, make_age_bin_age_group_max_dict

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


DIARRHEA_EVENT_COUNT_COLS = make_cols_demographically_specific('diarrhea_event_count', 2, 5)
DIARRHEA_EVENT_COUNT_COLS.append('diarrhea_event_count')


class DiarrheaEtiologyState(State):
    """
    Sets up a diarrhea etiology state (e.g. a column that states that simulant
    either has diarrhea due to rotavirus or does not have diarrhea due to
    rotavirus). Child class of State.

    Parameters
    ----------
    state_id: str
        string that describes the etiology state.

    key: str
        key is a necessary input to ensure random numbers are generated
        correctly @Alecwd: can you help me define key better here?
    """
    def __init__(self, state_id, key='state'):
        State.__init__(self, state_id)

        self.state_id = state_id

        self.event_count_column = state_id + '_event_count'

    def setup(self, builder):
        columns = [self.event_count_column]

        self.population_view = builder.population_view(columns, 'alive')

        # @Alecwd: is this the best way to set up a population? I could
        # use a little help determining why the super is necessary below
        return super(DiarrheaEtiologyState, self).setup(builder)

    @listens_for('initialize_simulants')
    def load_population_columns(self, event):
        population_size = len(event.index)
        self.population_view.update(pd.DataFrame({self.event_count_column:
                                                 np.zeros(population_size)},
                                                 index=event.index))

    # Output metrics counting the number of cases of diarrhea and number of
    #     cases overall of diarrhea due to each pathogen
    @modifies_value('metrics')
    @uses_columns(DIARRHEA_EVENT_COUNT_COLS + [eti + '_event_count' for eti in ETIOLOGIES])
    def metrics(self, index, metrics, population_view):
        population = population_view.get(index)

        metrics[self.event_count_column] = population[self.event_count_column].sum()

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

    csmr_data: df
        df with csmr for each age, sex, year, loc

    mild_disability_weight: float
        disability weight associated with mild diarrhea

    modearte_disability_weight: float
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
        self.severity_dict = {}
        self.severity_dict["severe"] = severe_disability_weight
        self.severity_dict["moderate"] = moderate_disability_weight
        self.severity_dict["mild"] = mild_disability_weight
        self.mild_severity_split = mild_severity_split
        self.moderate_severity_split = moderate_severity_split
        self.severe_severity_split = severe_severity_split
        self.duration_data = duration_data

    def setup(self, builder):
        columns = ['diarrhea']
        self.population_view = builder.population_view(columns, 'alive')

        # create a lookup table and establish a source for excess mortality
        self.diarrhea_excess_mortality = builder.rate(
            'excess_mortality.diarrhea')
        self.diarrhea_excess_mortality.source = builder.lookup(
            self.excess_mortality_data)

        # create a lookup table and establish a source for duration
        self.duration = builder.value('duration.diarrhea')
        self.duration.source = builder.lookup(self.duration_data)

        # create a randomness stream
        self.randomness = builder.randomness('determine_diarrhea_severity')

    @listens_for('initialize_simulants')
    @uses_columns(['diarrhea', 'diarrhea_event_time', 'diarrhea_event_end_time'] + DIARRHEA_EVENT_COUNT_COLS)
    def create_columns(self, event):

        length = len(event.index)

        df = pd.DataFrame({'diarrhea':['healthy']*length}, index=event.index)

        df['diarrhea_event_count'] = pd.Series([0]*length, index=df.index)

        df['diarrhea_event_time'] = pd.Series([pd.NaT]*length, index=df.index)

        df['diarrhea_event_end_time'] = pd.Series([pd.NaT]*length,
                                                  index=df.index)

        event.population_view.update(df)

    # delete the diarrhea csmr from the background mortality rate
    @modifies_value('csmr_data')
    def csmr(self):
        return self.csmr_data

    @modifies_value('mortality_rate')
    @uses_columns(['diarrhea'], 'alive')
    def mortality_rates(self, index, rates_df, population_view):
        # @ Alecwd: would I want to use a population_view passed in by the
        #     mortality_rates method or the population_view established
        #     earlier in this class (self.population_view)? Does it matter?
        population = population_view.get(index)

        # only apply excess mortality to people with severe diarrhea
        rates_df['death_due_to_severe_diarrhea'] = (
            self.diarrhea_excess_mortality(population.index, skip_post_processor=True)
            * (population['diarrhea'] == 'severe_diarrhea'))

        return rates_df

    @modifies_value('disability_weight')
    def disability_weight(self, index):
        population = self.population_view.get(index)
        disability_weights = pd.Series(np.zeros(len(index), dtype=float), index=index)
        # Mild, moderate, and severe each have their own disability weight,
        #     which we assign in the loop below.
        # In the future, we may want pathogens to be differentially
        #     associated with severity
        for severity in ["mild", "moderate", "severe"]:
            severity_index = population.query("diarrhea == '{}_diarrhea'".format(severity)).index
            disability_weights[severity_index] = self.severity_dict[severity]
        return disability_weights



    # FIXME: This is a super slow function. Try to speed it up by using numbers
    #     instead of strings
    # TODO: This method needs some more tests. Open to suggestions on how to
    #    best test this method
    @listens_for('time_step', priority=6)
    @uses_columns(['diarrhea', 'diarrhea_event_time', 'age', 'sex'] +
                  ETIOLOGIES +
                  [eti + '_event_count' for eti in ETIOLOGIES] +
                  DIARRHEA_EVENT_COUNT_COLS, 'alive and diarrhea == "healthy"')
    def move_people_into_diarrhea_state(self, event):
        """
        Determines who should move from the healthy state to the diarrhea state
        and counts both cases of diarrhea and cases of diarrhea due to specific
        etiologies
        """
        # Now we're making it so that only healthy people can get diarrhea
        #     (i.e. people currently with diarrhea are not susceptible for
        #     reinfection). This is the assumption were working with for
        #     now, but we may want to change in the future so that people
        #     currently infected with diarrhea can be reinfected
        pop = event.population

        # for people that got diarrhea due to an etiology (or multiple
        #     etiologies) in the current time step, we manually set the
        #     diarrhea column to equal "diarrhea"
        for etiology in ETIOLOGIES:
            pop.loc[pop['{}'.format(etiology)] == etiology, 'diarrhea'] = 'diarrhea'
            pop.loc[pop['{}'.format(etiology)] == etiology, '{}_event_count'.format(etiology)] += 1

        # now we want to make sure we're counting the bouts of diarrhea
        #    correctly, for each specific age/sex/year. We need demographic-
        #    specific counts for the incidence rates that we'll calculate later
        affected_pop = pop.query("diarrhea == 'diarrhea'")

        # key= age_bin, and value=age_bin_max
        age_bin_age_group_max_dict = make_age_bin_age_group_max_dict(age_group_id_min=2,
                                                                     age_group_id_max=5)

        current_year = pd.Timestamp(event.time).year

        # also track the overall count among all simulants in the simulation
        affected_pop['diarrhea_event_count'] += 1

        # set diarrhea event time
        affected_pop['diarrhea_event_time'] = pd.Timestamp(event.time)

        # get diarrhea severity splits
        mild_weight = self.mild_severity_split
        moderate_weight = self.moderate_severity_split
        severe_weight = self.severe_severity_split

        # Now we split out diarrhea by severity split. We use the choice method
        #    CEAM.framework.randomness. This is probably the simplest way of
        #    assigning assigning severity splits and we need to decide if it
        #    is the right way
        affected_pop['diarrhea'] = self.randomness.choice(affected_pop.index,
                                          ["mild_diarrhea", "moderate_diarrhea", "severe_diarrhea"],
                                          [mild_weight, moderate_weight, severe_weight])

        event.population_view.update(affected_pop)


    # TODO: Confirm whether or not we need different durations for different
    #     severity levels
    # TODO: Per conversation with Abie on 2.22, we would like to have a
    #     distribution surrounding duration
    @uses_columns(['diarrhea', 'diarrhea_event_time', 'diarrhea_event_end_time'] + \
                  ETIOLOGIES, 'alive and diarrhea != "healthy"')
    @listens_for('time_step', priority=8)
    def apply_remission(self, event):

        affected_population = event.population

        # TODO: I want to think of another test for apply_remission.
        #     There was an error before (event.index instead of
        #     affected_population.index was being passed in). Alec/James:
        #     any suggestions for another test for apply_remission?
        if not affected_population.empty:
            duration_series = pd.to_timedelta(self.duration(affected_population.index),
                                                            unit='D')

            affected_population['diarrhea_event_end_time'] = duration_series + \
                                                             affected_population['diarrhea_event_time']

            # manually set diarrhea to healthy and set all etiology columns to
            #     healthy as well
            current_time = pd.Timestamp(event.time)

            affected_population.loc[affected_population['diarrhea_event_end_time'] <= current_time, 'diarrhea'] = 'healthy'

            for etiology in ETIOLOGIES:
                affected_population['{}'.format(etiology)] = 'healthy'

        event.population_view.update(affected_population[ETIOLOGIES +
                                     ['diarrhea', 'diarrhea_event_end_time']])


def diarrhea_factory():
    """
    Factory that moves people from an etiology state to the diarrhea state and
        uses functions above to apply excess mortality and remission
    """
    list_of_modules = []

    # TODO: This seems like an easy place to make a mistake. The better way of
    #    getting the risk id data would be to run a get_ids query and have that
    #    return the ids we want (that statement could apply to anywhere we use
    #    a gbd id of some sort)
    dict_of_etiologies_and_eti_risks = {'cholera': 173,
                                        'other_salmonella': 174,
                                        'shigellosis': 175, 'EPEC': 176,
                                        'ETEC': 177, 'campylobacter': 178,
                                        'amoebiasis': 179,
                                        'cryptosporidiosis': 180,
                                        'rotaviral_entiritis': 181,
                                        'aeromonas': 182,
                                        'clostridium_difficile': 183,
                                        'norovirus': 184, 'adenovirus': 185,
                                        'unattributed_diarrhea': 'unattributed'}

    for pathogen, risk_id in dict_of_etiologies_and_eti_risks.items():

        module = DiseaseModel(pathogen)

        healthy = State('healthy', key=pathogen)

        # @Alecwd does it make sense to have the state_id and key be the same
        #    string?
        etiology_state = DiarrheaEtiologyState(pathogen,
                                               key=pathogen)

        etiology_specific_incidence = get_etiology_specific_incidence(
            eti_risk_id=risk_id, cause_id=302, me_id=1181)

        transition = RateTransition(etiology_state,
                                    pathogen,
                                    etiology_specific_incidence)

        healthy.transition_set.append(transition)
        healthy.allow_self_transitions()
        module.states.extend([healthy, etiology_state])

        list_of_modules.append(module)

    excess_mortality = get_severe_diarrhea_excess_mortality()

    time_step = config.simulation_parameters.time_step

    diarrhea_burden = DiarrheaBurden(excess_mortality_data=excess_mortality,
                                     csmr_data=get_cause_specific_mortality(1181),
                                     mild_disability_weight=get_disability_weight(healthstate_id=355),
                                     moderate_disability_weight=get_disability_weight(healthstate_id=356),
                                     severe_disability_weight=get_disability_weight(healthstate_id=357),
                                     mild_severity_split=get_severity_splits(1181, 2608),
                                     moderate_severity_split=get_severity_splits(1181, 2609),
                                     severe_severity_split=get_severity_splits(1181, 2610),
                                     duration_data=get_duration_in_days(1181))

    return list_of_modules + [diarrhea_burden]

