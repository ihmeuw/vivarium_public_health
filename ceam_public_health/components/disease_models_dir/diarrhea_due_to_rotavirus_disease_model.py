from ceam import config
from ceam.framework.state_machine import Transition, State, TransitionSet
from ceam_public_health.components.test_disease import DiseaseModel, DiseaseState, ExcessMortalityState, RateTransition, ProportionTransition, RemissionRateTransition, DiarrheaState
from ceam_inputs import get_etiology_specific_prevalence, get_etiology_specific_incidence, get_remission, get_excess_mortality, get_cause_specific_mortality
from ceam_inputs.gbd_ms_functions import get_disability_weight
from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns
import pandas as pd
import numpy as np
from ceam.framework.values import modifies_value
from datetime import timedelta


# list_of_etiologies = ['diarrhea_due_to_shigellosis', 'diarrhea_due_to_cholera', 'diarrhea_due_to_other_salmonella', 'diarrhea_due_to_EPEC', 'diarrhea_due_to_ETEC', 'diarrhea_due_to_campylobacter', 'diarrhea_due_to_amoebiasis', 'diarrhea_due_to_cryptosporidiosis', 'diarrhea_due_to_rotaviral_entiritis', 'diarrhea_due_to_aeromonas', 'diarrhea_due_to_clostridium_difficile', 'diarrhea_due_to_norovirus', 'diarrhea_due_to_adenovirus']
list_of_etiologies = ['diarrhea_due_to_norovirus', 'diarrhea_due_to_rotaviral_entiritis', 'diarrhea_due_to_adenovirus']


class EtiologyState(State):
    def __init__(self, state_id, prevalence_data, key='state'):
        State.__init__(self, state_id)

        self.prevalence_data = prevalence_data

 
    def name(self):
        return '{} ({}, {})'.format(self.state_id, self.prevalence_data)


class DiarrheaEtiologyState(EtiologyState):
    def __init__(self, state_id, prevalence_data, excess_mortality_data, disability_weight, dwell_time, key='state'):
        EtiologyState.__init__(self, state_id, prevalence_data)

        self.state_id = state_id

        self.excess_mortality_data = excess_mortality_data

        self._disability_weight = disability_weight

        self.event_count_column = state_id + '_event_count'

        # TODO: Use remission to get a dwell time
        self.dwell_time = dwell_time

        if isinstance(self.dwell_time, timedelta):
            self.dwell_time = self.dwell_time.total_seconds()

    def setup(self, builder):
        columns = [self.state_id, 'diarrhea', self.event_count_column]

        self.population_view = builder.population_view(columns, 'alive')

        self.clock = builder.clock()

        self.mortality = builder.rate('excess_mortality.diarrhea')
        self.mortality.source = builder.lookup(self.excess_mortality_data)

        # TODO: Figure out what the line below is doing
        return super(DiarrheaEtiologyState, self).setup(builder)

 
    @listens_for('initialize_simulants')
    def load_population_columns(self, event):
        population_size = len(event.index)
        self.population_view.update(pd.DataFrame({self.event_count_column: np.zeros(population_size)}, index=event.index))

    # FIXME: Think we're correctly counting the number of cases, though it would be worthwhile to restrict people from getting diarrhea due to a pathogen while they already are in the diarrhea state
    def _transition_side_effect(self, index):
        pop = self.population_view.get(index)

        pop = pop.query('diarrhea == "healthy"')

        pop[self.event_count_column] += 1

        pop['diarrhea_event_time'] = self.clock().timestamp()

        self.population_view.update(pop)


    def name(self):
        return '{} ({}, {})'.format(self.state_id, self.parent_cause_id, self.prevalence_data)

   
    @modifies_value('metrics')
    @uses_columns(['diarrhea_event_count'] + [i + '_event_count' for i in list_of_etiologies])
    def metrics(self, index, metrics, population_view):
        population = population_view.get(index)

        metrics[self.event_count_column] = population[self.event_count_column].sum()
        metrics['diarrhea_event_count'] = population['diarrhea_event_count'].sum()

        return metrics


    @modifies_value('disability_weight')
    def disability_weight(self, index):
        population = self.population_view.get(index)
        return self._disability_weight * (population['diarrhea'] == 'diarrhea')


class handle_diarrhea_excess_mortality():
    def __init__(self, excess_mortality_data):
        self.excess_mortality_data = excess_mortality_data

    def setup(self, builder):
        columns = ['diarrhea']
        self.population_view = builder.population_view(columns, 'alive')
        self.mortality = builder.rate('excess_mortality.diarrhea')
        self.mortality.source = builder.lookup(self.excess_mortality_data)
    
    @modifies_value('mortality_rate')
    @uses_columns(['diarrhea'], 'alive')
    def mortality_rates(self, index, rates, population_view):
        population = self.population_view.get(index)

        return rates + self.mortality(population.index) * (population['diarrhea'] == 'diarrhea')


class handle_remission():
    def __init__(self, remission_data):
        self.remission_data = remission_data


    def setup(self, builder):
        self.clock = builder.clock()

        self.duration = timedelta(days=5).total_seconds()


    @uses_columns(['diarrhea', 'diarrhea_event_time'] + list_of_etiologies)
    @listens_for('time_step')
    def _apply_remission(self, event):

        population = event.population_view.get(event.index)

        # manually set diarrhea to healthy and set all etiology columns to healthy as well

        affected_population = population.query("diarrhea == 'diarrhea'").copy()

        affected_population.loc[affected_population['diarrhea_event_time'] < self.clock().timestamp() - self.duration, 'diarrhea'] = 'healthy'

        for etiology in list_of_etiologies:
            affected_population.loc[affected_population.diarrhea == 'healthy', '{}'.format(etiology)] = 'healthy'
       
        event.population_view.update(affected_population[list_of_etiologies + ['diarrhea']]) 


def test_diarrhea_factory():

    list_of_modules = []

    states_dict = {}

    transition_dict = {}
 
    # dict_of_etiologies_and_eti_risks = {'cholera': 173, 'other_salmonella': 174, 'shigellosis': 175, 'EPEC': 176, 'ETEC': 177, 'campylobacter': 178, 'amoebiasis': 179, 'cryptosporidiosis': 180, 'rotaviral_entiritis': 181, 'aeromonas': 182, 'clostridium_difficile': 183, 'norovirus': 184, 'adenovirus': 185}

    dict_of_etiologies_and_eti_risks = {'adenovirus': 185, 'rotaviral_entiritis': 181, 'norovirus': 184}

    for key, value in dict_of_etiologies_and_eti_risks.items():

        diarrhea_due_to_pathogen = 'diarrhea_due_to_{}'.format(key)

        # TODO -- what does this module do for us?
        module = DiseaseModel(diarrhea_due_to_pathogen) 

        # TODO: Where should I define the healthy state?
        healthy = State('healthy', key= diarrhea_due_to_pathogen)

        etiology_specific_prevalence = get_etiology_specific_prevalence(eti_risk_id=value, cause_id=302, me_id=1181)

        # TODO: Get severity split draws so that we can have full uncertainty surrounding disability
        # Potential FIXME: Might want to actually have severity states in the future. Will need to figure out how to make sure that people with multiple pathogens have only one severity
        etiology_state = DiarrheaEtiologyState(diarrhea_due_to_pathogen, key=diarrhea_due_to_pathogen, prevalence_data=etiology_specific_prevalence, excess_mortality_data=get_excess_mortality(1181), disability_weight=0.2319, dwell_time=timedelta(days=30.5))
        # risk=rota cause=diarrhea me_id=diarrhea

        etiology_specific_incidence = get_etiology_specific_incidence(eti_risk_id=value, cause_id=302, me_id=1181)

        transition = RateTransition(etiology_state,
                                    'incidence_rate.diarrhea_due_to_{}'.format(key),
                                    etiology_specific_incidence)

        healthy.transition_set.append(transition)

        # TODO: After the MVS is finished, include transitions to non-fully healthy states (e.g. malnourished and stunted health states)
        # TODO: Figure out how remission rates can be different across diarrhea due to the different etiologies

        module.states.extend([healthy, etiology_state])

        list_of_modules.append(module)


    @listens_for('initialize_simulants')
    @uses_columns(['diarrhea', 'diarrhea_event_count', 'diarrhea_event_time'])
    def _create_diarrhea_column(event):

        length = len(event.index)

        diarrhea_series = pd.Series(['healthy'] * length)
        falses = np.zeros((length, 1), dtype=int)

        df = pd.DataFrame(falses, columns=['diarrhea_event_count'])
        df['diarrhea'] = diarrhea_series

        df['diarrhea_event_time'] = 0 

        event.population_view.update(df)


    @listens_for('time_step')
    @uses_columns(['diarrhea', 'diarrhea_event_count'] + list_of_etiologies)
    def _establish_diarrhea_excess_mortality_state(event):

        pop = event.population_view.get(event.index)

        # TODO: Confirm assumption that people who have diarrhea are not at risk while they have diarrhea
        pop = pop.query("diarrhea == 'healthy'")

        for etiology in list_of_etiologies:

            pop.loc[pop['{}'.format(etiology)] == etiology, 'diarrhea'] = 'diarrhea'

        pop.loc[pop['diarrhea'] == 'diarrhea', 'diarrhea_event_count'] += 1

        event.population_view.update(pop[['diarrhea', 'diarrhea_event_count']])


    excess_mort = handle_diarrhea_excess_mortality(get_excess_mortality(1181))

    remission = handle_remission(get_remission(1181))

    list_of_module_and_functs = list_of_modules + [_establish_diarrhea_excess_mortality_state, _create_diarrhea_column, excess_mort, remission]

    return list_of_module_and_functs


# End.
