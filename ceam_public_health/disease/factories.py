from ceam_inputs import get_disability_weight, get_prevalence, get_excess_mortality

from .state import DiseaseState, ExcessMortalityState, TransientDiseaseState, ProportionTransition


def make_disease_state(cause, side_effect_function=None):
    prevalence = get_prevalence(cause)
    disability_weight = get_disability_weight(cause)
    excess_mortality = get_excess_mortality(cause)

    if excess_mortality:
        return ExcessMortalityState(cause.name,
                                    prevalence_data=prevalence,
                                    disability_weight=disability_weight,
                                    excess_mortality_data=excess_mortality,
                                    side_effect_function=side_effect_function)
    return DiseaseState(cause.name,
                        prevalence_data=prevalence,
                        disability_weight=disability_weight,
                        side_effect_function=side_effect_function)


def make_severity_splits(cause):
    root_state = TransientDiseaseState(cause.name, track_events=False)
    splits = cause.severity_splits

    mild_state = make_disease_state(splits.mild)
    root_state.add_transition(mild_state, proportion=splits.mild.split)
    moderate_state = make_disease_state(splits.moderate)
    root_state.add_transition(moderate_state, proportion=splits.moderate.split)
    severe_state = make_disease_state(splits.severe)
    root_state.add_transition(severe_state, proportion=splits.severe.split)

    states = [root_state, mild_state, moderate_state, severe_state]

    if splits.asymptomatic:
        asymptomatic_state = make_disease_state(splits.asymptomatic)
        root_state.add_transition(asymptomatic_state, proportion=splits.asymptomatic.split)
        states.append(asymptomatic_state)

    return states

