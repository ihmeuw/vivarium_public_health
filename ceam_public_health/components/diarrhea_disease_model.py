import pandas as pd
import numpy as np
import operator
from datetime import timedelta

from ceam import config
from ceam.framework.state_machine import State
from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns
from ceam.framework.values import modifies_value
from ceam.framework.randomness import choice

from ceam_inputs import get_severity_splits
from ceam_inputs import get_severe_diarrhea_excess_mortality
from ceam_inputs import get_age_bins
from ceam_inputs.gbd_ms_functions import get_disability_weight
from ceam_inputs import (get_etiology_specific_prevalence,
                         get_etiology_specific_incidence, get_duration_in_days,
                         get_excess_mortality, get_cause_specific_mortality)
from ceam_inputs import make_age_group_1_to_4_rates_constant

from ceam_public_health.components.disease import DiseaseModel, RateTransition
from ceam_public_health.components.util import (make_cols_demographically_specific,
                                                make_age_bin_age_group_max_dict)
from ceam_public_health.components.accrue_susceptible_person_time import (
    AccrueSusceptiblePersonTime)


list_of_etiologies = ['diarrhea_due_to_shigellosis',
                      'diarrhea_due_to_cholera',
                      'diarrhea_due_to_other_salmonella',
                      'diarrhea_due_to_EPEC', 'diarrhea_due_to_ETEC',
                      'diarrhea_due_to_campylobacter',
                      'diarrhea_due_to_amoebiasis',
                      'diarrhea_due_to_cryptosporidiosis',
                      'diarrhea_due_to_rotaviral_entiritis',
                      'diarrhea_due_to_aeromonas',
                      'diarrhea_due_to_clostridium_difficile',
                      'diarrhea_due_to_norovirus',
                      'diarrhea_due_to_adenovirus',
                      'diarrhea_due_to_unattributed']


DIARRHEA_EVENT_COUNT_COLS = make_cols_demographically_specific('diarrhea_event_count', 2, 5)
DIARRHEA_EVENT_COUNT_COLS.append('diarrhea_event_count')


# TODO: Update doc string
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
    @uses_columns(DIARRHEA_EVENT_COUNT_COLS + [i + '_event_count' for i in
                                               list_of_etiologies])
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
        4) determines when a simulant should remit out of the diarrhea state

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

    duration_data: df
        df with duration data (in days) for each age, sex, year, loc
    """
    def __init__(self, excess_mortality_data, csmr_data,
                 mild_disability_weight, moderate_disability_weight,
                 severe_disability_weight, duration_data):
        self.excess_mortality_data = excess_mortality_data
        self.csmr_data = csmr_data
        self.severity_dict = {}
        self.severity_dict["severe"] = severe_disability_weight
        self.severity_dict["moderate"] = moderate_disability_weight
        self.severity_dict["mild"] = mild_disability_weight
        self.duration_data = duration_data

    def setup(self, builder):
        columns = ['diarrhea']
        self.population_view = builder.population_view(columns, 'alive')

        # create a rate table and establish a source for excess mortality
        self.diarrhea_excess_mortality = builder.rate(
            'excess_mortality.diarrhea')
        self.diarrhea_excess_mortality.source = builder.lookup(
            self.excess_mortality_data)

        self.clock = builder.clock()

        # create a rate table and establish a source for duration
        self.duration = builder.value('duration.diarrhea')
        self.duration.source = builder.lookup(self.duration_data)

    # delete the diarrhea csmr from the background mortality rate
    @modifies_value('csmr_data')
    def mmeids(self):
        return self.csmr_data

    @modifies_value('mortality_rate')
    @uses_columns(['diarrhea'], 'alive')
    def mortality_rates(self, index, rates_df, population_view):
        # @ Alecwd: would I want to use a population_view passed in by the
        #     mortality_rates method or the population_view established
        #     earlier in this class (self.population_view)? Does it matter?
        population = population_view.get(index)

        # only apply excess mortality to people with severe diarrhea
        rates_df['death_due_to_severe_diarrhea'] = self.diarrhea_excess_mortality(
            population.index, skip_post_processor=True) * \
                (population['diarrhea'] == 'severe_diarrhea')

        return rates_df

    @modifies_value('disability_weight')
    def disability_weight(self, index):
        population = self.population_view.get(index)

        # Initialize a series where each value is 0.
        #     We add in disability to people in the infected states below
        dis_weight_series = pd.Series(0, index=index)

        # Assert error if the diarrhea column has values that we do not expect
        assert set(population.diarrhea.unique()).issubset(['healthy',
                                                           'mild_diarrhea',
                                                           'moderate_diarrhea',
                                                           'severe_diarrhea']), \
            "simulants can have no, mild, moderate, or severe diarrhea" + \
            " this assert statement is meant to confirm that there are" + \
            " no values outside of what we expect"

        # Mild, moderate, and severe each have their own disability weight,
        #     which we assign in the loop below.
        # In the future, we may want pathogens to be differentially
        #     associated with severity
        for severity in ["mild", "moderate", "severe"]:
            severity_index = population.query("diarrhea == '{}_diarrhea'".format(severity)).index
            dis_weight_series.loc[severity_index] = self.severity_dict[severity]

        return dis_weight_series


    # TODO: Shorten the length of this function
    # FIXME: This is a super slow function. Try to speed it up by using numbers
    #     instead of strings
    # TODO: Might be worthwhile to have code read from top to bottom in the
    #    order of priority
    # TODO: This method needs some more tests. Open to suggestions on how to
    #    best test this method
    @listens_for('time_step', priority=6)
    @uses_columns(['diarrhea', 'diarrhea_event_time', 'age', 'sex'] +
                  list_of_etiologies +
                  [i + '_event_count' for i in list_of_etiologies] +
                  DIARRHEA_EVENT_COUNT_COLS, 'alive')
    def move_people_into_diarrhea_state(self, event):
        """
        Determines who should move from the healthy state to the diarrhea state
        and counts both cases of diarrhea and cases of diarrhea due to specific
        etiologies
        """

        pop = event.population_view.get(event.index)

        # Now we're making it so that only healthy people can get diarrhea
        #     (i.e. people currently with diarrhea are not susceptible for
        #     reinfection). This is the assumption were working with for
        #     now, but we may want to change in the future so that people
        #     currently infected with diarrhea can be reinfected
        pop = pop.query("diarrhea == 'healthy'")

        # for people that got diarrhea due to an etiology (or multiple
        #     etiologies) in the current time step, we manually set the
        #     diarrhea column to equal "diarrhea"
        for etiology in list_of_etiologies:
            pop.loc[pop['{}'.format(etiology)] == etiology, 'diarrhea'] = 'diarrhea'
            pop.loc[pop['{}'.format(etiology)] == etiology, '{}_event_count'.format(etiology)] += 1

        # now we want to make sure we're counting the bouts of diarrhea
        #    correctly, for each specific age/sex/year. We need demographic-
        #    specific counts for the incidence rates that we'll calculate later
        affected_pop = pop.query("diarrhea == 'diarrhea'")

        # key= age_bin, and value=age_bin_max
        age_bin_age_group_max_dict = make_age_bin_age_group_max_dict(age_group_id_min=2,
                                                                     age_group_id_max=5)

        last_age_group_max = 0
        current_year = pd.Timestamp(event.time).year

        for sex in ["Male", "Female"]:
            for age_bin, upr_bound in age_bin_age_group_max_dict:
                # We use GTE age group lower bound and LT age group upper bound
                #     because of how GBD age groups are set up. For example, a
                #     A simulant can be 1 or 4.999 years old and be considered
                #     part of the 1-5 year old group, but once they turn 5 they
                #     are part of the 5-10 age group
                affected_pop.loc[(affected_pop['age'] < upr_bound) &
                                 (affected_pop['age'] >= last_age_group_max) &
                                 (affected_pop['sex'] == sex),
                                 'diarrhea_event_count_{a}_in_year_{c}_among_{s}s'.format(
                                 a=age_bin, c=current_year, s=sex)] += 1
                last_age_group_max = upr_bound

        # also track the overall count among all simulants in the simulation
        affected_pop['diarrhea_event_count'] += 1

        # set diarrhea event time
        affected_pop['diarrhea_event_time'] = pd.Timestamp(event.time)

        # get diarrhea severity splits
        mild_weight = get_severity_splits(1181, 2608)
        moderate_weight = get_severity_splits(1181, 2609)
        severe_weight = get_severity_splits(1181, 2610)

        # Now we split out diarrhea by severity split. We use the choice method
        #    CEAM.framework.randomness. This is probably the simplest way of
        #    assigning assigning severity splits and we need to decide if it
        #    is the right way
        affected_pop['diarrhea'] = choice('determine_diarrhea_severity',
                                          affected_pop.index,
                                          ["mild_diarrhea", "moderate_diarrhea", "severe_diarrhea"],
                                          [mild_weight, moderate_weight, severe_weight])

        event.population_view.update(affected_pop)


    # TODO: Confirm whether or not we need different durations for different
    #     severity levels
    # TODO: Per conversation with Abie on 2.22, we would like to have a
    #     distribution surrounding duration
    @uses_columns(['diarrhea', 'diarrhea_event_time', 'diarrhea_event_end_time'] + \
                  list_of_etiologies, 'alive')
    @listens_for('time_step', priority=8)
    def apply_remission(self, event):

        population = event.population_view.get(event.index)

        affected_population = population.query("diarrhea != 'healthy'").copy()

        # TODO: I want to think of another test for apply_remission.
        #     There was an error before (event.index instead of
        #     affected_population.index was being passed in). Alec/James: 
        #     any suggestions for another test for apply_remission?
        affected_population['duration'] = pd.to_timedelta(self.duration(
                                                          affected_population.index),
                                                          unit='D')

        affected_population['diarrhea_event_end_time'] = affected_population['duration'] + \
                                                         affected_population['diarrhea_event_time']

        # manually set diarrhea to healthy and set all etiology columns to
        #     healthy as well
        current_time = pd.Timestamp(event.time)

        affected_population.loc[affected_population['diarrhea_event_end_time'] <= current_time, 'diarrhea'] = 'healthy'

        for etiology in list_of_etiologies:
            affected_population['{}'.format(etiology)] = 'healthy'

        event.population_view.update(affected_population[list_of_etiologies +
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
                                        'unattributed': 'unattributed'}

    for key, value in dict_of_etiologies_and_eti_risks.items():

        diarrhea_due_to_pathogen = 'diarrhea_due_to_{}'.format(key)

        module = DiseaseModel(diarrhea_due_to_pathogen)

        healthy = State('healthy', key=diarrhea_due_to_pathogen)

        # @Alecwd does it make sense to have the state_id and key be the same
        #    string?
        etiology_state = DiarrheaEtiologyState(diarrhea_due_to_pathogen,
                                               key=diarrhea_due_to_pathogen)

        etiology_specific_incidence = get_etiology_specific_incidence(
            eti_risk_id=value, cause_id=302, me_id=1181)

        # TODO: Need to figure out how to change priority on a RateTransition
        #     so that we can get ors_clock working
        transition = RateTransition(etiology_state,
                                    diarrhea_due_to_pathogen,
                                    etiology_specific_incidence)

        healthy.transition_set.append(transition)

        module.states.extend([healthy, etiology_state])

        list_of_modules.append(module)

    @listens_for('initialize_simulants')
    @uses_columns(['diarrhea', 'diarrhea_event_time', 'diarrhea_event_end_time'] + DIARRHEA_EVENT_COUNT_COLS)
    def create_columns(event):

        length = len(event.index)

        df = pd.DataFrame({'diarrhea':['healthy']*length}, index=event.index)

        for col in DIARRHEA_EVENT_COUNT_COLS:
            df[col] = pd.Series([0]*length, index=df.index)

        df['diarrhea_event_time'] = pd.Series([pd.NaT]*length, index=df.index)

        df['diarrhea_event_end_time'] = pd.Series([pd.NaT]*length,
                                                  index=df.index)

        event.population_view.update(df)



    excess_mortality = get_severe_diarrhea_excess_mortality()

    diarrhea_burden = DiarrheaBurden(excess_mortality_data=excess_mortality,
                                     csmr_data=get_cause_specific_mortality(1181),
                                     mild_disability_weight=get_disability_weight(healthstate_id=355),
                                     moderate_disability_weight=get_disability_weight(healthstate_id=356),
                                     severe_disability_weight=get_disability_weight(healthstate_id=357),
                                     duration_data=get_duration_in_days(1181))

    list_of_module_and_functs = list_of_modules + [create_columns,
                                                   diarrhea_burden,
                                                   # TODO: Will want to move
                                                   #    AccrueSusceptiblePersonTime
                                                   #    so that it's only
                                                   #    calculated in GBD years
                                                   AccrueSusceptiblePersonTime(
                                                       "diarrhea",
                                                       "severe_diarrhea")]

    return list_of_module_and_functs
