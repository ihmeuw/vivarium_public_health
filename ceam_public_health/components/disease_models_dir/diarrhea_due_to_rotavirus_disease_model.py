from ceam import config
from ceam.framework.state_machine import Transition, State, TransitionSet
from ceam_public_health.components.disease import DiseaseModel, DiseaseState, ExcessMortalityState, RateTransition, ProportionTransition, RemissionRateTransition, DiarrheaState
from ceam_inputs import get_etiology_specific_prevalence, get_etiology_specific_incidence, get_remission, get_excess_mortality, get_cause_specific_mortality
from ceam_inputs.gbd_ms_functions import get_disability_weight
from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns


# FIXME: Instead of using the factory for diarrhea, use objects to make sure that the code is fully organized.
# When you want a simulation component that is nested, return a list of nested components within the setup phase

# TODO: Figure out how to handle lack of prevalence data, since prevalence data is a required argument for ExcessMortalityState
# but not needed for DiarrheaExcessMortalityState
class DiarrheaExcessMortalityState(ExcessMortalityState):
    def setup(self, builder):

        delattr(ExcessMortalityState, prevalence_data) #TODO: Might want a design change. This breaks SPL rules. http://stackoverflow.com/questions/6057130/python-deleting-a-class-attribute-in-a-subclass
        
        super(DiarrheaExcessMortalityState, self).setup(builder)
        # FIXME: how to handle diarrhea randomness
        self.random = builder.randomness("diarrhea")

    @listens_for('initialize_simulants')
    @uses_columns(['diarrhea'])
    def _create_diarrhea_colum(self, event):
        length = len(event.index)
        falses = np.zeros((length, 1), dtype=bool)
        df = pd.DataFrame(falses, columns=['diarrhea'])
        
        event.population_view.update(df)

    # TODO: Determine if this should happen at the prepare stage and what priority it should be given
    @listens_for('time_step__prepare')
    @uses_columns(['diarrhea', 'diarrhea_due_to_rotavirus'])
    def _establish_diarrhea_excess_mortality_state(self, event, population_view):
        population = self.population_view.get(population.index) #TODO: Better way of determining who has diarrhea?
        affected_population = population.query('diarrhea_due_to_rotavirus == True').copy() # or diarrhea_due_to_salmonella == True, etc.
        if not affected_population.empty:
            affected_population['diarrhea'] = True
            
        # TODO: Put in a set trace to make sure this is working correctly

        event.population_view.update(affected_population, index=affected_population.index)


class DiarrheaEtiologyState(State):
    def __init__(self, state_id, prevalence_data, key='state'):
        State.__init__(self, state_id)

        self.prevalence_data = prevalence_data
    
    # def setup(self, builder):
        # not sure what needs to go into the setup
        # super(State, self).setup(builder)
        # FIXME: how to handle diarrhea randomness
        # self.random = builder.randomness(key)
    
    def name(self):
        return '{} ({}, {})'.format(self.state_id, self.prevalence_data)

  
def diarrhea_factory():
    
    module = DiseaseModel('diarrhea_due_to_rotavirus')

    
    healthy = State('healthy', key='diarrhea_due_to_rotavirus')

    diarrhea_due_to_rotavirus = DiarrheaEtiologyState('diarrhea_due_to_rotavirus', key='diarrhea_due_to_rotavirus', prevalence_data=get_etiology_specific_prevalence(eti_risk_id=181, # risk=rota
                                                                                                                                                             cause_id=302, me_id=1181)) # cause=diarrhea me_id=diarrhea

    diarrhea_due_to_rotavirus_transition = RateTransition(diarrhea_due_to_rotavirus, 
                                                                   'diarrhea_due_to_rotavirus', 
                                                                   get_etiology_specific_incidence(eti_risk_id=181, # risk=rota
                                                                                                   cause_id=302, me_id=1181)) # cause=diarrhea me_id=diarrhea
    
    healthy.transition_set.extend([diarrhea_due_to_rotavirus_transition])


    # TODO: Make states (using State class) for diarrhea due to the different pathogens. Then create an excess mortality state for anyone that is in any of the diarrhea states
    # FIXME: Might be a little strange if someone has diarrhea due to different pathogens but has multiple severities. might want to do severity split post diarrhea assignment
    diarrhea = DiarrheaExcessMortalityState('diarrhea',
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

    module.states.extend([healthy, diarrhea_due_to_rotavirus])

    return module


# End.
