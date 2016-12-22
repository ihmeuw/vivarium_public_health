from ceam import config
from ceam.framework.state_machine import Transition, State, TransitionSet
from ceam_public_health.components.disease import DiseaseModel, DiseaseState, ExcessMortalityState, RateTransition, ProportionTransition, RemissionRateTransition, DiarrheaState
from ceam_inputs import get_etiology_and_severity_specific_prevalence, get_etiology_and_severity_specific_incidence, get_remission, get_excess_mortality, get_cause_specific_mortality, get_diarrhea_severity_split_excess_mortality 
from ceam_inputs.gbd_ms_functions import get_disability_weight

# FIXME: Instead of using the factory for diarrhea, use objects to make sure that the code is fully organized.
# When you want a simulation component that is nested, return a list of nested components within the setup phase
  
def diarrhea_factory():
    
    module = DiseaseModel('severe_diarrhea_due_to_rotavirus')

    
    healthy = State('healthy', key='severe_diarrhea_due_to_rotavirus')

    # TODO: Make states (using State class) for diarrhea due to the different pathogens. Then create an excess mortality state for anyone that is in any of the diarrhea states
    # FIXME: Might be a little strange if someone has diarrhea due to different pathogens but has multiple severities. might want to do severity split post diarrhea assignment
    severe_diarrhea_due_to_rotavirus = ExcessMortalityState('severe_diarrhea_due_to_rotavirus', 
                                                     disability_weight=get_disability_weight(2610),
                                                     excess_mortality_data=get_diarrhea_severity_split_excess_mortality(get_excess_mortality(1181), 'severe'),
                                                     # FIXME: Since severe diarrhea accounts for all diarrhea mortality, think it's ok to just bring in cause-specific mortality rate here. Don't need to get csmr for mild/moderate.
                                                     cause_specific_mortality_data=get_cause_specific_mortality(1181),
                                                     prevalence_data=get_etiology_and_severity_specific_prevalence(eti_risk_id=181, # risk=rota
                                                                                                     cause_id=302, me_id=2610)) # cause=diarrhea me_id=severe diarrhea
    
    severe_diarrhea_due_to_rotavirus_transition = RateTransition(severe_diarrhea_due_to_rotavirus, 
                                                                   'severe_diarrhea_due_to_rotavirus', 
                                                                   get_etiology_and_severity_specific_incidence(eti_risk_id=181, # risk=rota
                                                                                                   cause_id=302, me_id=2610)) # cause=diarrhea me_id=severe diarrhea
    
    healthy.transition_set.extend([severe_diarrhea_due_to_rotavirus_transition])

    # TODO: After the MVS is finished, include transitions to non-fully healthy states (e.g. malnourished and stunted health states)
    # TODO: Figure out how remission rates can be different across diarrhea due to the different etiologies
    remission_transition = RateTransition(healthy, 'healthy', get_remission(1181))

    severe_diarrhea_due_to_rotavirus.transition_set.append(Transition(healthy))

    module.states.extend([healthy, severe_diarrhea_due_to_rotavirus])

    return module


# End.
