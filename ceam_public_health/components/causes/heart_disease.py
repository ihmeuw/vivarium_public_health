from ceam import config
from ceam.framework.state_machine import Transition, State, TransitionSet

from ceam_inputs import get_incidence, make_gbd_disease_state
from ceam_inputs.gbd_ms_functions import get_post_mi_heart_failure_proportion_draws, get_angina_proportions, get_asympt_ihd_proportions, load_data_from_cache
from ceam_inputs.util import gbd_year_range
from ceam_inputs.gbd_mapping import causes

from ceam_public_health.components.disease import DiseaseModel, RateTransition, ProportionTransition
from ceam_public_health.components.healthcare_access import hospitalization_side_effect_factory


def factory():
    module = DiseaseModel('ihd')

    healthy = State('healthy', key='ihd')

    location_id = config.simulation_parameters.location_id
    year_start, year_end = gbd_year_range()

    heart_attack = make_gbd_disease_state(causes.heart_attack, dwell_time=28, side_effect_function=hospitalization_side_effect_factory(0.6, 0.7, 'heart attack')) #rates as per Marcia e-mail 1/19/17
    mild_heart_failure = make_gbd_disease_state(causes.mild_heart_failure)
    moderate_heart_failure = make_gbd_disease_state(causes.moderate_heart_failure)
    severe_heart_failure = make_gbd_disease_state(causes.severe_heart_failure)

    asymptomatic_angina = make_gbd_disease_state(causes.asymptomatic_angina)

    mild_angina = make_gbd_disease_state(causes.mild_angina)
    moderate_angina = make_gbd_disease_state(causes.moderate_angina)
    severe_angina = make_gbd_disease_state(causes.severe_angina)

    asymptomatic_ihd = make_gbd_disease_state(causes.asymptomatic_ihd)

    heart_attack_transition = RateTransition(heart_attack, 'heart_attack', get_incidence(causes.heart_attack.incidence))
    healthy.transition_set.append(heart_attack_transition)

    heart_failure_buckets = TransitionSet(allow_null_transition=False, key="heart_failure_split")
    heart_failure_buckets.extend([
        ProportionTransition(mild_heart_failure, proportion=0.182074),
        ProportionTransition(moderate_heart_failure, proportion=0.149771),
        ProportionTransition(severe_heart_failure, proportion=0.402838),
        ])

    angina_buckets = TransitionSet(allow_null_transition=False, key="angina_split")
    angina_buckets.extend([
        ProportionTransition(asymptomatic_angina, proportion=0.304553),
        ProportionTransition(mild_angina, proportion=0.239594),
        ProportionTransition(moderate_angina, proportion=0.126273),
        ProportionTransition(severe_angina, proportion=0.32958),
        ])
    healthy.transition_set.append(RateTransition(angina_buckets, 'non_mi_angina', get_incidence(causes.angina_not_due_to_MI.incidence)))

    heart_attack.transition_set.allow_null_transition=False

    # TODO: Need to figure out best way to implemnet functions here
    # TODO: Need to figure out where transition from rates to probabilities needs to happen
    hf_prop_df = load_data_from_cache(get_post_mi_heart_failure_proportion_draws, col_name='proportion', src_column='draw_{draw}', location_id=location_id, year_start=year_start, year_end=year_end)
    angina_prop_df = load_data_from_cache(get_angina_proportions, col_name='proportion', src_column='angina_prop')#, year_start=year_start, year_end=year_end)
    asympt_prop_df = load_data_from_cache(get_asympt_ihd_proportions, col_name='proportion', src_column='asympt_prop_{draw}', location_id=location_id, year_start=year_start, year_end=year_end)

    # post-mi transitions
    # TODO: Figure out if we can pass in me_id here to get incidence for the correct cause of heart failure
    # TODO: Figure out how to make asymptomatic ihd be equal to whatever is left after people get heart failure and angina
    heart_attack.transition_set.append(ProportionTransition(heart_failure_buckets, proportion=hf_prop_df))
    heart_attack.transition_set.append(ProportionTransition(angina_buckets, proportion=angina_prop_df))
    heart_attack.transition_set.append(ProportionTransition(asymptomatic_ihd, proportion=asympt_prop_df))

    mild_heart_failure.transition_set.append(heart_attack_transition)
    moderate_heart_failure.transition_set.append(heart_attack_transition)
    severe_heart_failure.transition_set.append(heart_attack_transition)
    asymptomatic_angina.transition_set.append(heart_attack_transition)
    mild_angina.transition_set.append(heart_attack_transition)
    moderate_angina.transition_set.append(heart_attack_transition)
    severe_angina.transition_set.append(heart_attack_transition)
    asymptomatic_ihd.transition_set.append(heart_attack_transition)

    module.states.extend([healthy, heart_attack, mild_heart_failure, moderate_heart_failure, severe_heart_failure, asymptomatic_angina, mild_angina, moderate_angina, severe_angina, asymptomatic_ihd])
    return module
