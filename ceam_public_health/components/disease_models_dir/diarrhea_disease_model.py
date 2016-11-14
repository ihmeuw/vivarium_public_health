from ceam import config
from ceam.framework.state_machine import Transition, State, TransitionSet
from ceam_public_health.components.disease import DiseaseModel, DiseaseState, ExcessMortalityState, IncidenceRateTransition, ProportionTransition, RemissionRateTransition, DiarrheaState
from ceam_inputs.gbd_ms_functions import get_disability_weight

def diarrhea_factory():
    """Hello world for diarrhea cost effectiveness analysis"""
    module = DiseaseModel('diarrhea')

    # initialize an object of the State class. object has 2 attributes, state_id and transition_set
    healthy = State('healthy', key='diarrhea')

    # TODO: Need to determine where to put code to aeteological split
    # TODO: Need to employ severity splits (mild, moderate, and severe diarrhea) in the future 
    # FIXME: Figure out what to use for the disability weight, currently using dis weight draws for moderate diarrhea
    diarrhea = DiarrheaState('diarrhea', disability_weight=get_disability_weight(dis_weight_modelable_entity_id=2609), modelable_entity_id=1181, prevalence_meid=1181) 

    diarrhea_transition = IncidenceRateTransition(diarrhea, 'diarrhea', modelable_entity_id=1181)

    healthy.transition_set.extend([diarrhea_transition])
  
    # TODO: After the MVS is finished, include transitions to non-fully healthy states (e.g. malnourished and stunted health states)
    remission_transition = RemissionRateTransition(healthy, 'healthy', modelable_entity_id=1181)

    diarrhea.transition_set.append(Transition(healthy))

    module.states.extend([healthy, diarrhea])

    return module


# End.
