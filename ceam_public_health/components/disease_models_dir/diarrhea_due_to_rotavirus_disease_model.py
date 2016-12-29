from ceam import config
from ceam.framework.state_machine import Transition, State, TransitionSet
from ceam_public_health.components.disease import DiseaseModel, DiseaseState, ExcessMortalityState, RateTransition, ProportionTransition, RemissionRateTransition, DiarrheaState
from ceam_inputs import get_etiology_specific_prevalence, get_etiology_specific_incidence, get_remission, get_excess_mortality, get_cause_specific_mortality
from ceam_inputs.gbd_ms_functions import get_disability_weight

# FIXME: Instead of using the factory for diarrhea, use objects to make sure that the code is fully organized.
# When you want a simulation component that is nested, return a list of nested components within the setup phase
  
def diarrhea_factory():
    
    module = DiseaseModel('diarrhea_due_to_rotavirus')

    
    healthy = State('healthy', key='diarrhea_due_to_rotavirus')

    # TODO: Make states (using State class) for diarrhea due to the different pathogens. Then create an excess mortality state for anyone that is in any of the diarrhea states
    # FIXME: Might be a little strange if someone has diarrhea due to different pathogens but has multiple severities. might want to do severity split post diarrhea assignment
    diarrhea_due_to_rotavirus = ExcessMortalityState('diarrhea_due_to_rotavirus', 
                                                     # TODO: Need to figure out what to do with disability weights. If we're going to model severity splits, each severity will get its own disability weight. if not, we need to use the severity split draws to get 1k draws of weighted disability weight.
                                                     disability_weight=.24 * get_disability_weight(2608) + .62 * get_disability_weight + .14 * get_disability_weight(2610),
                                                     excess_mortality_data=get_excess_mortality(1181),
                                                     cause_specific_mortality_data=get_cause_specific_mortality(1181),
                                                     prevalence_data=get_etiology_specific_prevalence(eti_risk_id=181, # risk=rota
                                                                                                     cause_id=302, me_id=1181)) # cause=diarrhea me_id=diarrhea
    
    diarrhea_due_to_rotavirus_transition = RateTransition(diarrhea_due_to_rotavirus, 
                                                                   'diarrhea_due_to_rotavirus', 
                                                                   get_etiology_specific_incidence(eti_risk_id=181, # risk=rota
                                                                                                   cause_id=302, me_id=1181)) # cause=diarrhea me_id=diarrhea
    
    healthy.transition_set.extend([diarrhea_due_to_rotavirus_transition])

    # TODO: After the MVS is finished, include transitions to non-fully healthy states (e.g. malnourished and stunted health states)
    # TODO: Figure out how remission rates can be different across diarrhea due to the different etiologies
    remission_transition = RateTransition(healthy, 'healthy', get_remission(1181))

    diarrhea_due_to_rotavirus.transition_set.append(Transition(healthy))

    module.states.extend([healthy, diarrhea_due_to_rotavirus])

    return module


# End.
