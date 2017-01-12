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

#class DiarrheaExcessMortalityState(ExcessMortalityState):
#    # Overwrite some of the functions from ExcessMortalityState
#    def setup1(self, builder):
#        columns = [self.condition, self.state_id]
#        if self.event_count_column:
#            columns += [self.event_count_column]
#        self.population_view = builder.population_view(columns, 'alive')
        
#    def setup(self, builder):
#        self.mortality = builder.rate('excess_mortality.{}'.format(self.state_id))
#        self.mortality.source = builder.lookup(self.excess_mortality_data)
#        return self.setup(builder)

#    @modifies_value('mortality_rate')
#    def mortality_rates(self, index, rates):
#        population = self.population_view.get(index)

#        return rates + self.mortality(population.index) * (population[self.state_id] == self.state_id)


class DiarrheaEtiologyState(State):
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

    
    # TODO: Determine if this should happen at the prepare stage and what priority it should be given
    @listens_for('time_step')
    @uses_columns(['diarrhea', 'diarrhea_due_to_rotavirus', 'diarrhea_event_count'])
    def _establish_diarrhea_excess_mortality_state(self, event):
        index = event.population_view.manager._population.query("diarrhea_due_to_rotavirus != 'healthy'").index
        affected_population = event.population_view.get(index).copy()
        # get(population.index) #TODO: Better way of determining who has diarrhea?
        # affected_population = population.query('diarrhea_due_to_rotavirus == True').copy() # or diarrhea_due_to_salmonella == True, etc.
        if not affected_population.empty:
            affected_population['diarrhea'] = 'diarrhea'
            affected_population['diarrhea_event_count'] += 1 

        event.population_view.update(affected_population)

        # import pdb; pdb.set_trace()   
 
    def name(self):
        return '{} ({}, {})'.format(self.state_id, self.prevalence_data)

    # @modifies_value('metrics')
    # def metrics(self, event, index, metrics):
    #    population = event.population_view.get(index)
    #    metrics['diarrhea_event_count'] = population['diarrhea_event_count'].sum()
    #    return metrics

  
def diarrhea_factory():
    
    module = DiseaseModel('diarrhea_due_to_rotavirus')

    
    healthy = State('healthy', key='diarrhea_due_to_rotavirus')

    diarrhea_due_to_rotavirus = DiarrheaEtiologyState('diarrhea_due_to_rotavirus', key='diarrhea_due_to_rotavirus', prevalence_data=get_etiology_specific_prevalence(eti_risk_id=181, # risk=rota
                                                                                                                                                             cause_id=302, me_id=1181)) # cause=diarrhea me_id=diarrhea

    diarrhea_due_to_rotavirus_transition = RateTransition(diarrhea_due_to_rotavirus, 
                                                                   'incidence_rate.diarrhea_due_to_rotavirus', 
                                                                   get_etiology_specific_incidence(eti_risk_id=181, # risk=rota
                                                                                                   cause_id=302, me_id=1181)) # cause=diarrhea me_id=diarrhea
    
    healthy.transition_set.extend([diarrhea_due_to_rotavirus_transition])


    # TODO: Make states (using State class) for diarrhea due to the different pathogens. Then create an excess mortality state for anyone that is in any of the diarrhea states
    # FIXME: Might be a little strange if someone has diarrhea due to different pathogens but has multiple severities. might want to do severity split post diarrhea assignment
    diarrhea = ExcessMortalityState('diarrhea',
                                                     # TODO: Get severity split draws so that we can have full uncertainty surrounding disability
                                                     # Potential FIXME: Might want to actually have severity states in the future
                                                     disability_weight=.24 * get_disability_weight(2608) + .62 * get_disability_weight(2609) + .14 * get_disability_weight(2610),
                                                     excess_mortality_data=get_excess_mortality(1181),
                                                     cause_specific_mortality_data=get_cause_specific_mortality(1181),
                                                     prevalence_data=None)

    # TODO: After the MVS is finished, include transitions to non-fully healthy states (e.g. malnourished and stunted health states)
    # TODO: Figure out how remission rates can be different across diarrhea due to the different etiologies
    remission_transition = RateTransition(healthy, 'healthy', get_remission(1181))

    diarrhea_due_to_rotavirus.transition_set.append(Transition(healthy))

    module.states.extend([healthy, diarrhea_due_to_rotavirus, diarrhea])

    return module


# End.
