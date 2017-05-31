from ceam_inputs import get_incidence, make_gbd_disease_state, causes

from ceam_public_health.components.disease import DiseaseModel
from ceam_public_health.components.healthcare_access import hospitalization_side_effect_factory


def factory():
    healthy = DiseaseState('healthy', track_events=False, key='all_stroke')
    healthy.allow_self_transitions()
    # TODO: need to model severity splits for stroke. then we can bring in correct disability weights (dis weights
    # correspond to healthstate ids which correspond to sequela) 
    hemorrhagic_stroke = make_disease_state(causes.hemorrhagic_stroke,
                                            dwell_time=28,
                                            side_effect_function=hospitalization_side_effect_factory(0.52, 0.6, 'hemorrhagic stroke')) #rates as per Marcia e-mail
    ischemic_stroke = make_disease_state(causes.ischemic_stroke,
                                         dwell_time=28,
                                         side_effect_function=hospitalization_side_effect_factoryhospitalization_side_effect_factory(0.52, 0.6, 'ischemic stroke')) #rates as per Marcia e-mail
    chronic_stroke = make_disease_state(causes.chronic_stroke)

    healthy.add_transition(hemorrhagic_stroke, rates=get_incidence(causes.hemorrhagic_stroke.incidence))
    healthy.add_transition(ischemic_stroke, rates=get_incidence(causes.ischemic_stroke.incidence))

    hemorrhagic_stroke.add_transition(chronic_stroke)
    ischemic_stroke.add_transition(chronic_stroke)

    return DiseaseModel('all_stroke',
                        states=[healthy, hemorrhagic_stroke, ischemic_stroke, chronic_stroke])
