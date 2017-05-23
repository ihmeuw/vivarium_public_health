from ceam import config
from ceam.framework.randomness import filter_for_probability
from ceam.framework.event import emits, Event
from ceam.framework.state_machine import Transition, State, TransitionSet

from ceam.framework.population import uses_columns

from ceam_public_health.components.disease import DiseaseModel, RateTransition, ProportionTransition, make_disease_state
from ceam_inputs import get_incidence
from ceam_inputs.gbd_ms_functions import get_post_mi_heart_failure_proportion_draws, get_angina_proportions, get_asympt_ihd_proportions, load_data_from_cache

from ceam_inputs.gbd_mapping import causes


def side_effect_factory(male_probability, female_probability, hospitalization_type):
    @emits('hospitalization')
    @uses_columns(['sex'])
    def hospitalization_side_effect(index, emitter, population_view):
        pop = population_view.get(index)
        pop['probability'] = 0.0
        pop.loc[pop.sex == 'Male', 'probability'] = male_probability
        pop.loc[pop.sex == 'Female', 'probability'] = female_probability
        effective_population = filter_for_probability('Hospitalization due to {}'.format(hospitalization_type), pop.index, pop.probability)
        new_event = Event(effective_population)
        emitter(new_event)
    return hospitalization_side_effect


def heart_disease_factory():
    module = DiseaseModel('ihd')

    healthy = State('healthy', key='ihd')

    # Calculate an adjusted disability weight for the acute heart attack phase that
    # accounts for the fact that our timestep is longer than the phase length
    # TODO: This doesn't account for the fact that our timestep is longer than 28 days
    timestep = config.simulation_parameters.time_step
    weight = 0.43*(2/timestep) + 0.07*(28/timestep)

    heart_attack = make_disease_state(causes.heart_attack,
                                      dwell_time=28,
                                      side_effect_function=side_effect_factory(0.6, 0.7, 'heart attack')) #rates as per Marcia e-mail 1/19/17

    mild_heart_failure = make_disease_state(causes.mild_heart_failure)
    moderate_heart_failure = make_disease_state(causes.moderate_heart_failure)
    severe_heart_failure = make_disease_state(causes.severe_heart_failure)

    asymptomatic_angina = make_disease_state(causes.asymptomatic_angina)
    mild_angina = make_disease_state(causes.mild_angina)
    moderate_angina = make_disease_state(causes.moderate_angina)
    severe_angina = make_disease_state(causes.severe_angina)

    asymptomatic_ihd = make_disease_state(causes.asymptomatic_ihd)

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
    hf_prop_df = get_post_mi_heart_failure_proportion_draws()
    angina_prop_df = get_angina_proportions()
    asympt_prop_df = get_asympt_ihd_proportions()

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


def stroke_factory():
    module = DiseaseModel('all_stroke')

    healthy = State('healthy', key='all_stroke')
    # TODO: need to model severity splits for stroke. then we can bring in correct disability weights (dis weights
    # correspond to healthstate ids which correspond to sequela) 
    hemorrhagic_stroke = make_disease_state(causes.hemorrhagic_stroke, dwell_time=28, side_effect_function=side_effect_factory(0.52, 0.6, 'hemorrhagic stroke')) #rates as per Marcia e-mail
    ischemic_stroke = make_disease_state(causes.ischemic_stroke, dwell_time=28, side_effect_function=side_effect_factory(0.52, 0.6, 'ischemic stroke')) #rates as per Marcia e-mail
    chronic_stroke = make_disease_state(causes.chronic_stroke)

    hemorrhagic_transition = RateTransition(hemorrhagic_stroke, 'hemorrhagic_stroke', get_incidence(causes.hemorrhagic_stroke.incidence))
    ischemic_transition = RateTransition(ischemic_stroke, 'ischemic_stroke', get_incidence(causes.ischemic_stroke.incidence))
    healthy.transition_set.extend([hemorrhagic_transition, ischemic_transition])

    hemorrhagic_stroke.transition_set.append(Transition(chronic_stroke))
    ischemic_stroke.transition_set.append(Transition(chronic_stroke))

    chronic_stroke.transition_set.extend([hemorrhagic_transition, ischemic_transition])

    module.states.extend([healthy, hemorrhagic_stroke, ischemic_stroke, chronic_stroke])

    return module
