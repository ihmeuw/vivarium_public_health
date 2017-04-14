from ceam.framework.state_machine import Transition, State
from ceam_inputs import get_incidence, make_gbd_disease_state
from ceam_inputs.gbd_mapping import causes

from ceam_public_health.components.disease import DiseaseModel, RateTransition
from ceam_public_health.components.healthcare_access import hospitalization_side_effect_factory

def factory():
    module = DiseaseModel('all_stroke')

    healthy = State('healthy', key='all_stroke')
    # TODO: need to model severity splits for stroke. then we can bring in correct disability weights (dis weights
    # correspond to healthstate ids which correspond to sequela) 
    hemorrhagic_stroke = make_gbd_disease_state(causes.hemorrhagic_stroke, dwell_time=28, side_effect_function=hospitalization_side_effect_factory(0.52, 0.6, 'hemorrhagic stroke')) #rates as per Marcia e-mail
    ischemic_stroke = make_gbd_disease_state(causes.ischemic_stroke, dwell_time=28, side_effect_function=hospitalization_side_effect_factory(0.52, 0.6, 'ischemic stroke')) #rates as per Marcia e-mail
    chronic_stroke = make_gbd_disease_state(causes.chronic_stroke)

    hemorrhagic_transition = RateTransition(hemorrhagic_stroke, 'hemorrhagic_stroke', get_incidence(causes.hemorrhagic_stroke.incidence))
    ischemic_transition = RateTransition(ischemic_stroke, 'ischemic_stroke', get_incidence(causes.ischemic_stroke.incidence))
    healthy.transition_set.extend([hemorrhagic_transition, ischemic_transition])

    hemorrhagic_stroke.transition_set.append(Transition(chronic_stroke))
    ischemic_stroke.transition_set.append(Transition(chronic_stroke))

    chronic_stroke.transition_set.extend([hemorrhagic_transition, ischemic_transition])

    module.states.extend([healthy, hemorrhagic_stroke, ischemic_stroke, chronic_stroke])

    return module
