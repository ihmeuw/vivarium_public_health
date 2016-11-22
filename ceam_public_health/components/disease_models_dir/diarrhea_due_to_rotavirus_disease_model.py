from ceam import config
from ceam.framework.state_machine import Transition, State, TransitionSet
from ceam_public_health.components.disease import DiseaseModel, DiseaseState, ExcessMortalityState, IncidenceRateTransition, ProportionTransition, RemissionRateTransition, DiarrheaState
from ceam_inputs.gbd_ms_functions import get_disability_weight

def diarrhea_factory():
    
    module = DiseaseModel('diarrhea_due_to_rotavirus')
    
    healthy = State('healthy', key='diarrhea_due_to_rotavirus')

    # TODO: Need to employ severity splits (mild, moderate, and severe diarrhea) in the future
    # FIXME: Figure out what to use for the disability weight, currently using dis weight draws for moderate diarrhea
    # TODO: Determine excess mortality for different severity levels of diarrhea. Are diarrheal cases due to different
    # etiologies different??
    # TODO: Gotta figure out how to determine prevalence of diarrhea due to specific 
    diarrhea_due_to_rotavirus = ExcessMortalityState('diarrhea_due_to_rotavirus', 
                                                     disability_weight=get_disability_weight(2609),
                                                     modelable_entity_id=1181, 
                                                     prevalence_meid=get_etiology_prevalence('rotavirus'))
    
    diarrhea_due_to_rotavirus_transition = IncidenceRateTransition(diarrhea_due_to_rotavirus, 
                                                                   'diarrhea_to_rotavirus', 
                                                                   get_etiology_incidence('rotavirus'))
    
    healthy.transition_set.extend([diarrhea_due_to_rotavirus_transition])

    # TODO: After the MVS is finished, include transitions to non-fully healthy states (e.g. malnourished and stunted health states)
    # TODO: Figure out how remission rates can be different across diarrhea due to the different etiologies
    remission_transition = RemissionRateTransition(healthy, 'healthy', modelable_entity_id=1181)

    diarrhea_due_to_rotavirus.transition_set.append(Transition(healthy))

    module.states.extend([healthy, diarrhea_due_to_rotavirus])

    return module


# End.
