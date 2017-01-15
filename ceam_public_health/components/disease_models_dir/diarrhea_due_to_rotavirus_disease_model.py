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

# FIXME: Instead of using the factory for diarrhea, use objects to make sure that the code is fully organized.
# When you want a simulation component that is nested, return a list of nested components within the setup phase

# TODO: Figure out how to handle lack of prevalence data, since prevalence data is a required argument for ExcessMortalityState
# but not needed for DiarrheaExcessMortalityState


class EtiologyState(State):
    def __init__(self, state_id, prevalence_data, key='state'):
        State.__init__(self, state_id)

        self.prevalence_data = prevalence_data
    

    @listens_for('initialize_simulants')
    @uses_columns(['diarrhea', 'diarrhea_event_count'])
    def _create_diarrhea_colum(self, event):
        length = len(event.index)
  
        diarrhea_series = pd.Series(['healthy'] * length)
        falses = np.zeros((length, 1), dtype=int)

        df = pd.DataFrame(falses, columns=['diarrhea_event_count'])
        df['diarrhea'] = diarrhea_series

        event.population_view.update(df)

    def name(self):
        return '{} ({}, {})'.format(self.state_id, self.prevalence_data)


class DiarrheaEtiologyState(EtiologyState):
    def __init__(self, state_id, parent_cause_id, prevalence_data, key='state'):
        EtiologyState.__init__(self, state_id, prevalence_data)

        self.state_id = state_id
        self.prevalence_data = prevalence_data
        self.parent_cause_id = parent_cause_id 
        self.parent_cause_id_column = self.parent_cause_id + '_event_count'

    def setup(self, builder):
        columns = [self.state_id]

        self.event_count_column = self.state_id + '_event_count'

        if self.event_count_column:
            columns += [self.event_count_column]

        self.population_view = builder.population_view(columns, 'alive')
        self.clock = builder.clock()

    # TODO: 3 functions below are copied exactly from Disease State. Figure out how to pull these exactly!
    @listens_for('initialize_simulants')
    def load_population_columns(self, event):
        population_size = len(event.index)
        self.population_view.update(pd.DataFrame({self.event_count_column: np.zeros(population_size)}, index=event.index))

    # TODO: Figure out what the 2 functs below are doing
    def next_state(self, index, population_view):
        eligible_index = index
        return super(DiarrheaEtiologyState, self).next_state(eligible_index, population_view)

    def _transition_side_effect(self, index):
        pop = self.population_view.get(index)

        pop[self.event_count_column] += 1

        self.population_view.update(pop)


    # TODO: Determine if this should happen at the prepare stage and what priority it should be given
    @listens_for('time_step')
    @uses_columns(['diarrhea', 'diarrhea_due_to_rotavirus', 'diarrhea_event_count'])
    def _establish_diarrhea_excess_mortality_state(self, event):
        affected_population = event.population_view.get(event.index).query("diarrhea_due_to_rotavirus != 'healthy'")

        if not affected_population.empty:
            affected_population['diarrhea'] = 'diarrhea'
            affected_population['diarrhea_event_count'] += 1 

        event.population_view.update(affected_population)

 
    def name(self):
        return '{} ({}, {})'.format(self.state_id, self.parent_cause_id, self.prevalence_data)

    
    @modifies_value('metrics')
    @uses_columns(['diarrhea_event_count', 'diarrhea_due_to_rotavirus_event_count'])
    def metrics(self, index, metrics, population_view):
        population = population_view.get(index)

        metrics[self.event_count_column] = population[self.event_count_column].sum()
        metrics[self.parent_cause_id_column] = population[self.parent_cause_id_column].sum()

        return metrics


def diarrhea_factory():
    
    module = DiseaseModel('diarrhea_due_to_rotavirus')
 
    healthy = State('healthy', key='diarrhea_due_to_rotavirus')

    diarrhea_due_to_rotavirus = DiarrheaEtiologyState('diarrhea_due_to_rotavirus', 'diarrhea', key='diarrhea_due_to_rotavirus', prevalence_data=get_etiology_specific_prevalence(eti_risk_id=181, cause_id=302, me_id=1181)) # risk=rota cause=diarrhea me_id=diarrhea

    diarrhea_due_to_rotavirus_transition = RateTransition(diarrhea_due_to_rotavirus, 
                                                                   'incidence_rate.diarrhea_due_to_rotavirus', 
                                                                   get_etiology_specific_incidence(eti_risk_id=181, # risk=rota
                                                                                                   cause_id=302, me_id=1181)) # cause=diarrhea me_id=diarrhea
    
    healthy.transition_set.extend([diarrhea_due_to_rotavirus_transition])


    # TODO: Make states (using State class) for diarrhea due to the different pathogens. Then create an excess mortality state for anyone that is in any of the diarrhea states
    # FIXME: Might be a little strange if someone has diarrhea due to different pathogens but has multiple severities. might want to do severity split post diarrhea assignment
    #diarrhea = ExcessMortalityState('diarrhea',
                                                     # TODO: Get severity split draws so that we can have full uncertainty surrounding disability
                                                     # Potential FIXME: Might want to actually have severity states in the future
    #                                                 disability_weight=.24 * get_disability_weight(2608) + .62 * get_disability_weight(2609) + .14 * get_disability_weight(2610),
     #                                                excess_mortality_data=get_excess_mortality(1181),
      #                                               cause_specific_mortality_data=get_cause_specific_mortality(1181),
       #                                              prevalence_data=None)

    # TODO: After the MVS is finished, include transitions to non-fully healthy states (e.g. malnourished and stunted health states)
    # TODO: Figure out how remission rates can be different across diarrhea due to the different etiologies
    remission_transition = RateTransition(healthy, 'healthy', get_remission(1181))

    diarrhea_due_to_rotavirus.transition_set.append(Transition(healthy))

    module.states.extend([healthy, diarrhea_due_to_rotavirus])

    return module


# End.
