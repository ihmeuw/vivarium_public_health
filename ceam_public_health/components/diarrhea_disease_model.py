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
from ceam_public_health.components.util import make_cols_demographically_specific, make_age_bin_age_group_max_dict
from ceam_public_health.components.accrue_susceptible_person_time import (
    AccrueSusceptiblePersonTime)

# FIXME: DIARRHEA IS NOT BEING DELETED IN THE CAUSE DELETED MORTALITY RATE! MUST FIX

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

diarrhea_event_count_cols = make_cols_demographically_specific('diarrhea_event_count', 2, 5)
diarrhea_event_count_cols.append('diarrhea_event_count')

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

    disability_weight: float
        disability associated with diarrhea, disability weight is not etiology specific
    """
    def __init__(self, state_id, key='state'):

        State.__init__(self, state_id)

        self.state_id = state_id

        self.event_count_column = state_id + '_event_count'

    def setup(self, builder):
        columns = [self.state_id, 'diarrhea', self.event_count_column]

        self.population_view = builder.population_view(columns, 'alive')

        # TODO: Determine if there is a better way to set up a population
        return super(DiarrheaEtiologyState, self).setup(builder)

    @listens_for('initialize_simulants')
    def load_population_columns(self, event):
        population_size = len(event.index)
        self.population_view.update(pd.DataFrame({self.event_count_column:
                                                 np.zeros(population_size)},
                                                 index=event.index))

    # Output metrics counting the number of cases of diarrhea and number of cases overall of diarrhea due to each pathogen
    # TODO: Why are the diarrhea event count cols in this uses_columns statement but they aren't being called? Are we counting them somewhere else?
    @modifies_value('metrics')
    @uses_columns(diarrhea_event_count_cols + [i + '_event_count' for i in
                                               list_of_etiologies])
    def metrics(self, index, metrics, population_view):
        population = population_view.get(index)

        metrics[self.event_count_column] = population[self.event_count_column].sum()

        return metrics

# FIXME: Docstring needs to be updated
# FIXME: This is where we want to make sure that cause-deleted mortality rate is correctly included in the model
#     Also include a test to ensure that diarrhea is being cause-deleted
# TODO: Eventually we may want to include transitions to non-fully healthy
#     states (e.g. malnourished and stunted health states)
# TODO: Eventually may want remission rates can be different across diarrhea due to
#     the different etiologies
class DiarrheaBurden:
    """
    Assigns an excess mortality and duration of diarrhea 
        to people that have diarrhea

    Parameters
    ----------
    excess_mortality_data: df
        df with excess mortality rate for each age, sex, year, loc

    cause_specific_mortality_data: df
        df with csmr for each age, sex, year, loc

    duration_data: df
        df with duration data (in days) for each age, sex, year, loc
    """
    def __init__(self, excess_mortality_data, cause_specific_mortality_data,
                 mild_disability_weight, moderate_disability_weight, 
                 severe_disability_weight, duration_data):
        self.excess_mortality_data = excess_mortality_data
        self.cause_specific_mortality_data = cause_specific_mortality_data
        self.severe_disability_weight = severe_disability_weight
        self.moderate_disability_weight = moderate_disability_weight
        self.mild_disability_weight = mild_disability_weight
        self.duration_data = duration_data

    def setup(self, builder):
        columns = ['diarrhea']
        self.population_view = builder.population_view(columns, 'alive')
        self.diarrhea_excess_mortality = builder.rate(
            'excess_mortality.diarrhea')
        self.diarrhea_excess_mortality.source = builder.lookup(
            self.excess_mortality_data)

        self.clock = builder.clock()

        self.duration = builder.value('duration.diarrhea')

        # this gives you a base value. intervention will change this value
        self.duration.source = builder.lookup(self.duration_data)

    @modifies_value('cause_specific_mortality_data')
    def mmeids(self):
        return self.cause_specific_mortality_data


    @modifies_value('mortality_rate')
    @uses_columns(['diarrhea'], 'alive')
    def mortality_rates(self, index, rates_df, population_view):
        # FIXME: Might want to use population_view passed in by function instead of
        #     self.population_view in line below
        population = self.population_view.get(index)
        # TODO: Need to write tests that ensure that only people with severe
        #     diarrhea have an elevated mortality. Ensure that people with mild
        #     and moderate diarrhea do not have an elevated mortality
        rates_df['death_due_to_severe_diarrhea'] = self.diarrhea_excess_mortality(
            population.index, skip_post_processor=True) * (population['diarrhea'] == 'severe_diarrhea')

        return rates_df

    # FIXME: Would be nice if functions with the @modifies_value('disability_weight') took an argument that is similar to the 'rates' argument in functions with the @modifies_value('mortality_rate') decorator
    # FIXME: Need to set a priority on this function so that it is set after
    #     _move_people_into_diarrhea_state
    # TODO: Does it make sense that this function doesn't get called if I change the name from disability weight to something else?
    @modifies_value('disability_weight')
    def disability_weight(self, index):
        population = self.population_view.get(index)

        # severe diarrhea
        dis_weight_series = self.severe_disability_weight * (population['diarrhea'] == 'severe_diarrhea')

        # moderate diarrhea
        moderate_index = population.query("diarrhea == 'moderate_diarrhea'").index
        dis_weight_series.loc[moderate_index] = self.moderate_disability_weight

        # mild diarrhea
        mild_index = population.query("diarrhea == 'mild_diarrhea'").index
        dis_weight_series.loc[mild_index] = self.mild_disability_weight 

        # TODO: Write a test to ensure that disability is only associated with
        #     severe diarrhea, and not mild/moderate
        return dis_weight_series

    # TODO: Confirm whether or not we need different durations for different
    #     severity levels
    # FIXME: Per conversation with Abie on 2.22, we would like to have a
    #     distribution surrounding duration
    # TODO: Confirm whether we need a distribution surrounding duration for this paper
    @uses_columns(['diarrhea', 'diarrhea_event_time', 'diarrhea_event_end_time'] + list_of_etiologies, 'alive')
    @listens_for('time_step', priority=8)
    def _apply_remission(self, event):

        population = event.population_view.get(event.index)

        affected_population = population.query("diarrhea != 'healthy'").copy()


        # TODO: Make this clearer. Why is event.index being used on a df that only includes people with diarrhea?
        affected_population['duration'] = pd.to_timedelta(self.duration(
                                                          event.index),
                                                          unit='D')

        affected_population['diarrhea_event_end_time'] = affected_population['duration'] + \
                                                         affected_population['diarrhea_event_time']

        # manually set diarrhea to healthy and set all etiology columns to
        #     healthy as well
        current_time = pd.Timestamp(event.time)

        affected_population.loc[affected_population['diarrhea_event_end_time'] <= current_time, 'diarrhea'] = 'healthy'

        # TODO: Confirm why we are setting etiologies to healthy here. Is this messing up counts?
        for etiology in list_of_etiologies:
            affected_population['{}'.format(etiology)] = 'healthy'

        # TODO: Confirm whether or not Duration column needs to be updated
        event.population_view.update(affected_population[list_of_etiologies + ['diarrhea', 'diarrhea_event_end_time']])


def diarrhea_factory():
    """
    Factory that moves people from an etiology state to the diarrhea state and
        uses functions above to apply excess mortality and remission
    """
    list_of_modules = []

    # TODO: Need to re-confirm that these are the correct rei_ids
    # TODO: This seems like an easy place to make a mistake. Is there a better way to set the risk ids for each etiology?
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

        # TODO: Where should I define the healthy state?
        healthy = State('healthy', key=diarrhea_due_to_pathogen)

        # TODO: Make sure it makes sense to have the state_id and key be the same thing
        etiology_state = DiarrheaEtiologyState(diarrhea_due_to_pathogen,
                                               key=diarrhea_due_to_pathogen)

        etiology_specific_incidence = get_etiology_specific_incidence(
            eti_risk_id=value, cause_id=302, me_id=1181)

        # FIXME: Merge in develop and use Alec's piecewise constant interpolation instead of the make_age_group_1_to_4_rates_constant function
        if config.getint('simulation_parameters', 'diarrhea_constant_incidence') == 1:
            etiology_specific_incidence = make_age_group_1_to_4_rates_constant(
                etiology_specific_incidence)

        # TODO: Need to figure out how to change priority on a RateTransition
        #     so that we can get ors_clock working
        transition = RateTransition(etiology_state,
                                    diarrhea_due_to_pathogen,
                                    etiology_specific_incidence)

        healthy.transition_set.append(transition)

        module.states.extend([healthy, etiology_state])

        list_of_modules.append(module)

    # TODO: Improve the name of this function
    # TODO: Put all of the columns into a dataframe and update the population_view one time
    @listens_for('initialize_simulants')
    @uses_columns(['diarrhea', 'diarrhea_event_time', 'diarrhea_event_end_time'] + diarrhea_event_count_cols)
    def _create_diarrhea_column(event):

        length = len(event.index)

        # TODO: Make one df, update one df as opposed to multiple one column
        #     updates
        event.population_view.update(pd.DataFrame({'diarrhea': ['healthy']*length},
                                                  index=event.index))

        for col in diarrhea_event_count_cols:
            event.population_view.update(pd.DataFrame({col: np.zeros(len(event.index),
                                                      dtype=int)}, index=event.index))

        event.population_view.update(pd.DataFrame({'diarrhea_event_time': [pd.NaT]*length},
                                                  index=event.index))
        event.population_view.update(pd.DataFrame({'diarrhea_event_end_time': [pd.NaT]*length},
                                                  index=event.index))


    # FIXME: This is a super slow function. Try to speed it up by using numbers
    #     instead of strings
    # TODO: I don't like how priorities are set in a different way than the code flows. Would be nice if priority 0 stuff could be at the top, priority 9 stuff at the bottom
    @listens_for('time_step', priority=6)
    @uses_columns(['diarrhea', 'diarrhea_event_time', 'age', 'sex'] + list_of_etiologies + [i + '_event_count' for i in list_of_etiologies] + diarrhea_event_count_cols, 'alive')
    def _move_people_into_diarrhea_state(event):
        """
        Determines who should move from the healthy state to the diarrhea state
        and counts both cases of diarrhea and cases of diarrhea due to specific
        etiologies
        """

        pop = event.population_view.get(event.index)

        # Potential FIXME: Now we're making it so that only healthy people can
        #     get diarrhea (i.e. people currently with diarrhea are not
        #     susceptible)
        pop = pop.query("diarrhea == 'healthy'")

        for etiology in list_of_etiologies:
            pop.loc[pop['{}'.format(etiology)] == etiology, 'diarrhea'] = 'diarrhea'
            pop.loc[pop['{}'.format(etiology)] == etiology, '{}_event_count'.format(etiology)] += 1

        last_age_group_max = 0

        # sort self.dict_of_age_group_name_and_max_values by value (max age)
        sorted_dict = make_age_bin_age_group_max_dict(age_group_id_min=2, age_group_id_max=5)
                            
        current_year = pd.Timestamp(event.time).year

        # need to set this up so that it counts events properly for specific
        #    age groups
        for sex in ["Male", "Female"]:
            for key, value in sorted_dict:
                pop.loc[(pop['diarrhea'] == 'diarrhea') & (pop['age'] < value) &
                        (pop['age'] >= last_age_group_max) & (pop['sex'] == sex),
                        'diarrhea_event_count_{k}_in_year_{c}_among_{s}s'.format(
                            k=key, c=current_year, s=sex)] += 1
                last_age_group_max = value

        pop.loc[pop['diarrhea'] == 'diarrhea', 'diarrhea_event_count'] += 1

        # set diarrhea event time here
        pop.loc[pop['diarrhea'] == 'diarrhea', 'diarrhea_event_time'] = pd.Timestamp(event.time)

        # FIXME: Why is the line below here? Doesn't seem to do anything. Should confirm it does nothing and then delete
        pop = pop.query("diarrhea == 'diarrhea'").copy()

        # get diarrhea severity splits
        mild_weight = get_severity_splits(1181, 2608)
        moderate_weight = get_severity_splits(1181, 2609)
        severe_weight = get_severity_splits(1181, 2610)

        pop['diarrhea'] = choice('determine_diarrhea_severity', pop.index,
                                 ["mild_diarrhea", "moderate_diarrhea",
                                  "severe_diarrhea"],
                                 [mild_weight, moderate_weight, severe_weight])

        event.population_view.update(pop[['diarrhea', 'diarrhea_event_time'] +
            [i + '_event_count' for i in list_of_etiologies] +
            diarrhea_event_count_cols])


    # TODO: Add some commenting letting the reader know that we're back into the factory
    excess_mortality = get_severe_diarrhea_excess_mortality()

    # if we want constant mortality, need to do some processing
    if config.getint('simulation_parameters', 'diarrhea_constant_mortality') == 1:
        excess_mortality = make_age_group_1_to_4_rates_constant(
            excess_mortality)

    # FIXME: Why is get_severity_splits being passed into the severe diarrhea_disability_weight function? That can't be right. There is also an error in the get_severity_splits code where draw_1 is hardcoded in. This will need to be updated.
    diarrhea_burden = DiarrheaBurden(excess_mortality_data=excess_mortality,
                                     cause_specific_mortality_data=get_cause_specific_mortality(1181),
                                     mild_disability_weight=get_disability_weight(2608),
                                     moderate_disability_weight=get_disability_weight(2609),
                                     severe_disability_weight=get_disability_weight(2610),
                                     duration_data=get_duration_in_days(1181))

    list_of_module_and_functs = list_of_modules + [_move_people_into_diarrhea_state,
                                                   _create_diarrhea_column,
                                                   diarrhea_burden, AccrueSusceptiblePersonTime(
						   "diarrhea", "severe_diarrhea")]

    return list_of_module_and_functs


# End.
