from ceam import config
from ceam.framework.state_machine import Transition, State, TransitionSet
from ceam_public_health.components.disease import DiseaseModel, RateTransition
from ceam_inputs import get_etiology_specific_prevalence, get_etiology_specific_incidence, get_duration_in_days, get_excess_mortality, get_cause_specific_mortality
from ceam_inputs.gbd_ms_functions import get_disability_weight
from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns
import pandas as pd
import numpy as np
from ceam.framework.values import modifies_value
from datetime import timedelta
from ceam_public_health.components.accrue_susceptible_person_time import AccrueSusceptiblePersonTime


list_of_etiologies = ['diarrhea_due_to_shigellosis', 'diarrhea_due_to_cholera', 'diarrhea_due_to_other_salmonella', 'diarrhea_due_to_EPEC', 'diarrhea_due_to_ETEC', 'diarrhea_due_to_campylobacter', 'diarrhea_due_to_amoebiasis', 'diarrhea_due_to_cryptosporidiosis', 'diarrhea_due_to_rotaviral_entiritis', 'diarrhea_due_to_aeromonas', 'diarrhea_due_to_clostridium_difficile', 'diarrhea_due_to_norovirus', 'diarrhea_due_to_adenovirus']


class DiarrheaEtiologyState(State):
    """
    Sets up a diarrhea etiology state (e.g. a column that states that simulant either has diarrhea due to rotavirus or does not have diarrhea due to rotavirus. Child class of State.    

    Parameters
    ----------
    state_id: str
        string that describes the etiology state. 

    disability_weight: float
        disability associated with diarrhea, regardless of which etiologies are associated with the bout of diarrhea
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
        self.population_view.update(pd.DataFrame({self.event_count_column: np.zeros(population_size)}, index=event.index))


    @modifies_value('metrics')
    @uses_columns(['diarrhea_event_count'] + [i + '_event_count' for i in list_of_etiologies])
    def metrics(self, index, metrics, population_view):
        population = population_view.get(index)

        metrics[self.event_count_column] = population[self.event_count_column].sum()
        metrics['diarrhea_event_count'] = population['diarrhea_event_count'].sum()

        return metrics


# TODO: After the MVS is finished, include transitions to non-fully healthy states (e.g. malnourished and stunted health states)
# TODO: Figure out how remission rates can be different across diarrhea due to the different etiologies
class DiarrheaBurden:
    """
    Assigns an excess mortality and duration to people that have diarrhea

    Parameters
    ----------
    excess_mortality_data: df
        df with excess mortality rate for each age, sex, year, loc

    cause_specific_mortality_data: df
        df with csmr for each age, sex, year, loc

    duration_data: df
        df with duration data (in days) for each age, sex, year, loc
    """
    def __init__(self, excess_mortality_data, cause_specific_mortality_data, disability_weight, duration_data):
        self.excess_mortality_data = excess_mortality_data
        self.cause_specific_mortality_data = cause_specific_mortality_data
        self._disability_weight = disability_weight
        self.duration_data = duration_data

    def setup(self, builder):
        columns = ['diarrhea']
        self.population_view = builder.population_view(columns, 'alive')
        self.mortality = builder.rate('excess_mortality.diarrhea')
        self.mortality.source = builder.lookup(self.excess_mortality_data)

        self.clock = builder.clock()

        self.duration = builder.value('duration.diarrhea')

        # this gives you a base value. intervention will change this value
        self.duration.source = builder.lookup(self.duration_data)

    @modifies_value('cause_specific_mortality_data')
    def mmeids(self):
        return self.cause_specific_mortality_data


    @modifies_value('mortality_rate')
    @uses_columns(['diarrhea'], 'alive')
    def mortality_rates(self, index, rates, population_view):
        population = self.population_view.get(index)

        return rates + self.mortality(population.index, skip_post_processor=True) * (population['diarrhea'] == 'diarrhea')


    @modifies_value('disability_weight')
    def disability_weight(self, index):
        population = self.population_view.get(index)
        return self._disability_weight * (population['diarrhea'] == 'diarrhea')


    # FIXME: Per conversation with Abie on 2.22, we would like to have a distribution surrounding duration
    @uses_columns(['diarrhea', 'diarrhea_event_time', 'diarrhea_event_end_time'] + list_of_etiologies)
    @listens_for('time_step', priority=8)
    def _apply_remission(self, event):

        population = event.population_view.get(event.index)

        affected_population = population.query("diarrhea == 'diarrhea'").copy()

        affected_population['duration'] = pd.to_timedelta(self.duration(event.index), unit='D')
        affected_population['diarrhea_event_end_time'] = affected_population['duration'] + affected_population['diarrhea_event_time']

        # manually set diarrhea to healthy and set all etiology columns to healthy as well
        current_time = pd.Timestamp(event.time)

        affected_population.loc[affected_population['diarrhea_event_end_time'] <= current_time, 'diarrhea'] = 'healthy'

        for etiology in list_of_etiologies:
            affected_population['{}'.format(etiology)] = 'healthy'

        event.population_view.update(affected_population[list_of_etiologies + ['diarrhea', 'diarrhea_event_end_time']])


def diarrhea_factory():
    """
    Factory that moves people from an etiology state to the diarrhea state and uses functions above to apply excess mortality and remission 
    """
    list_of_modules = []

    states_dict = {}

    transition_dict = {}
 
    dict_of_etiologies_and_eti_risks = {'cholera': 173, 'other_salmonella': 174, 'shigellosis': 175, 'EPEC': 176, 'ETEC': 177, 'campylobacter': 178, 'amoebiasis': 179, 'cryptosporidiosis': 180, 'rotaviral_entiritis': 181, 'aeromonas': 182, 'clostridium_difficile': 183, 'norovirus': 184, 'adenovirus': 185}

    for key, value in dict_of_etiologies_and_eti_risks.items():

        diarrhea_due_to_pathogen = 'diarrhea_due_to_{}'.format(key)

        module = DiseaseModel(diarrhea_due_to_pathogen) 

        # TODO: Where should I define the healthy state?
        healthy = State('healthy', key=diarrhea_due_to_pathogen)

        # TODO: Get severity split draws so that we can have full uncertainty surrounding disability
        # Potential FIXME: Might want to actually have severity states in the future. Will need to figure out how to make sure that people with multiple pathogens have only one severity
        etiology_state = DiarrheaEtiologyState(diarrhea_due_to_pathogen, key=diarrhea_due_to_pathogen)

        etiology_specific_incidence = get_etiology_specific_incidence(eti_risk_id=value, cause_id=302, me_id=1181)

        # TODO: Need to figure out how to change priority on a RateTransition so that we can get ors_clock working
        transition = RateTransition(etiology_state,
                                    diarrhea_due_to_pathogen,
                                    etiology_specific_incidence)

        healthy.transition_set.append(transition)

        module.states.extend([healthy, etiology_state])

        list_of_modules.append(module)


    @listens_for('initialize_simulants')
    @uses_columns(['diarrhea', 'diarrhea_event_count', 'diarrhea_event_time', 'diarrhea_event_end_time'])
    def _create_diarrhea_column(event):

        length = len(event.index)

        # TODO: Make one df, update one df as opposed to multiple one column updates
        event.population_view.update(pd.DataFrame({'diarrhea': ['healthy']*length}, index=event.index))
        event.population_view.update(pd.DataFrame({'diarrhea_event_count': np.zeros(len(event.index), dtype=int)}, index=event.index))

        event.population_view.update(pd.DataFrame({'diarrhea_event_time': [pd.NaT]*length}, index=event.index))
        event.population_view.update(pd.DataFrame({'diarrhea_event_end_time': [pd.NaT]*length}, index=event.index))

    @listens_for('time_step', priority=6)
    @uses_columns(['diarrhea', 'diarrhea_event_count', 'diarrhea_event_time'] + list_of_etiologies + [i + '_event_count' for i in list_of_etiologies])
    def _move_people_into_diarrhea_state(event):
        """
        Determines who should move from the healthy state to the diarrhea state and counts both cases of diarrhea and cases of diarrhea due to specific etiologies
        """
        
        pop = event.population_view.get(event.index)

        # Potential FIXME: Now we're making it so that only healthy people can get diarrhea (i.e. people currently with diarrhea are not susceptible)
        pop = pop.query("diarrhea == 'healthy'")

        for etiology in list_of_etiologies:
            # etiology_pop = pop.loc[pop['{}'.format(etiology)] == etiology]
            pop.loc[pop['{}'.format(etiology)] == etiology, 'diarrhea'] = 'diarrhea'
            pop.loc[pop['{}'.format(etiology)] == etiology, '{}_event_count'.format(etiology)] += 1

        pop.loc[pop['diarrhea'] == 'diarrhea', 'diarrhea_event_count'] += 1

        # set diarrhea event time here
        pop.loc[pop['diarrhea'] == 'diarrhea', 'diarrhea_event_time'] = pd.Timestamp(event.time)

        event.population_view.update(pop[['diarrhea', 'diarrhea_event_count', 'diarrhea_event_time'] + [i + '_event_count' for i in list_of_etiologies]])

    diarrhea_burden = DiarrheaBurden(excess_mortality_data=get_excess_mortality(1181), cause_specific_mortality_data=get_cause_specific_mortality(1181), disability_weight=0.2319, duration_data=get_duration_in_days(1181))

    list_of_module_and_functs = list_of_modules + [_move_people_into_diarrhea_state, _create_diarrhea_column, diarrhea_burden, AccrueSusceptiblePersonTime('diarrhea', 'diarrhea')]

    return list_of_module_and_functs


# End.
