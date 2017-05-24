from ceam import config
from ceam.framework.randomness import filter_for_probability
from ceam.framework.event import emits, Event
from ceam.framework.state_machine import Transition, State, TransitionSet

from ceam.framework.population import uses_columns

from ceam_public_health.components.disease import (DiseaseModel, RateTransition, DiseaseState, TransientDiseaseState,
                                                   ProportionTransition, make_disease_state)
from ceam_inputs import (get_incidence, get_post_mi_heart_failure_proportion_draws,
                         get_angina_proportions, get_asympt_ihd_proportions)

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
    # Calculate an adjusted disability weight for the acute heart attack phase that
    # accounts for the fact that our timestep is longer than the phase length
    # TODO: This doesn't account for the fact that our timestep is longer than 28 days
    timestep = config.simulation_parameters.time_step
    weight = 0.43*(2/timestep) + 0.07*(28/timestep)

    healthy = DiseaseState('healthy', track_events=False, key='ihd')
    healthy.allow_self_transitions()
    heart_attack = make_disease_state(causes.heart_attack,
                                      dwell_time=28,
                                      side_effect_function=side_effect_factory(0.6, 0.7, 'heart attack')) #rates as per Marcia e-mail 1/19/17

    heart_failure = TransientDiseaseState('heart_failure', track_events=False)
    mild_heart_failure = make_disease_state(causes.mild_heart_failure)
    moderate_heart_failure = make_disease_state(causes.moderate_heart_failure)
    severe_heart_failure = make_disease_state(causes.severe_heart_failure)

    angina = TransientDiseaseState('non_mi_angina', track_events=False)
    asymptomatic_angina = make_disease_state(causes.asymptomatic_angina)
    mild_angina = make_disease_state(causes.mild_angina)
    moderate_angina = make_disease_state(causes.moderate_angina)
    severe_angina = make_disease_state(causes.severe_angina)

    asymptomatic_ihd = make_disease_state(causes.asymptomatic_ihd)

    healthy.add_transition(heart_attack, rates=get_incidence(causes.heart_attack.incidence))
    heart_failure.add_transition(mild_heart_failure, proportion=0.182074)
    heart_failure.add_transition(moderate_heart_failure, proportion=0.149771)
    heart_failure.add_transition(severe_heart_failure, proportion=0.402838)

    healthy.add_transition(angina, rates=get_incidence(causes.angina_not_due_to_MI.incidence))
    angina.add_transition(asymptomatic_angina, proportion=0.304553)
    angina.add_transition(mild_angina, proportion=0.239594)
    angina.add_transition(moderate_angina, proportion=0.126273)
    angina.add_transition(severe_angina, proportion=0.32958)


    # TODO: Need to figure out best way to implement functions here
    # TODO: Need to figure out where transition from rates to probabilities needs to happen
    hf_prop_df = get_post_mi_heart_failure_proportion_draws()
    angina_prop_df = get_angina_proportions()
    asympt_prop_df = get_asympt_ihd_proportions()

    # post-mi transitions
    # TODO: Figure out if we can pass in me_id here to get incidence for the correct cause of heart failure
    # TODO: Figure out how to make asymptomatic ihd be equal to whatever is left after people get heart failure and angina
    heart_attack.add_transition(heart_failure, proportion=hf_prop_df)
    heart_attack.add_transition(angina, proportion=angina_prop_df)
    heart_attack.add_transition(asymptomatic_ihd, proportion=asympt_prop_df)

    mild_heart_failure.add_transition(heart_attack, rates=get_incidence(causes.heart_attack.incidence))
    moderate_heart_failure.add_transition(heart_attack, rates=get_incidence(causes.heart_attack.incidence))
    severe_heart_failure.add_transition(heart_attack, rates=get_incidence(causes.heart_attack.incidence))
    asymptomatic_angina.add_transition(heart_attack, rates=get_incidence(causes.heart_attack.incidence))
    mild_angina.add_transition(heart_attack, rates=get_incidence(causes.heart_attack.incidence))
    moderate_angina.add_transition(heart_attack, rates=get_incidence(causes.heart_attack.incidence))
    severe_angina.add_transition(heart_attack, rates=get_incidence(causes.heart_attack.incidence))
    asymptomatic_ihd.add_transition(heart_attack, rates=get_incidence(causes.heart_attack.incidence))

    return DiseaseModel('ihd',
                        states=[healthy,
                                heart_attack,
                                asymptomatic_ihd,
                                heart_failure, mild_heart_failure, moderate_heart_failure, severe_heart_failure,
                                angina, asymptomatic_angina, mild_angina, moderate_angina, severe_angina])


def stroke_factory():
    healthy = DiseaseState('healthy', track_events=False, key='all_stroke')
    healthy.allow_self_transitions()
    # TODO: need to model severity splits for stroke. then we can bring in correct disability weights (dis weights
    # correspond to healthstate ids which correspond to sequela) 
    hemorrhagic_stroke = make_disease_state(causes.hemorrhagic_stroke,
                                            dwell_time=28,
                                            side_effect_function=side_effect_factory(0.52, 0.6, 'hemorrhagic stroke')) #rates as per Marcia e-mail
    ischemic_stroke = make_disease_state(causes.ischemic_stroke,
                                         dwell_time=28,
                                         side_effect_function=side_effect_factory(0.52, 0.6, 'ischemic stroke')) #rates as per Marcia e-mail
    chronic_stroke = make_disease_state(causes.chronic_stroke)

    healthy.add_transition(hemorrhagic_stroke, rates=get_incidence(causes.hemorrhagic_stroke.incidence))
    healthy.add_transition(ischemic_stroke, rates=get_incidence(causes.ischemic_stroke.incidence))

    hemorrhagic_stroke.add_transition(chronic_stroke)
    ischemic_stroke.add_transition(chronic_stroke)

    return DiseaseModel('all_stroke',
                        states=[healthy, hemorrhagic_stroke, ischemic_stroke, chronic_stroke])
