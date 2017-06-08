from collections import namedtuple

from .data_transformations import (get_etiology_incidence, get_duration_in_days,
                                   get_severe_diarrhea_excess_mortality)

from ceam.framework.population import uses_columns

from ceam_inputs import (get_severity_splits, get_disability_weight,
                         get_cause_specific_mortality, causes, risk_factors)

from ceam_public_health.components.disease import (DiseaseModel, DiseaseState,
                                                   TransientDiseaseState, ExcessMortalityState)

Etiology = namedtuple('Etiology', ['name', 'model', 'recovery_transition', 'pre_trigger_state'])


def build_etiology_model(etiology_name, infection_side_effect=None):
    healthy = DiseaseState('healthy', track_events=False, key=etiology_name)
    sick = DiseaseState(etiology_name,
                        disability_weight=0,
                        side_effect_function=infection_side_effect)

    healthy.add_transition(sick, rates=get_etiology_incidence(etiology_name))
    healthy.allow_self_transitions()
    recovery_transition = sick.add_transition(healthy, triggered=True)

    return Etiology(name=etiology_name,
                    model=DiseaseModel(etiology_name, states=[healthy, sick]),
                    recovery_transition=recovery_transition,
                    pre_trigger_state=sick)


def build_diarrhea_model():
    healthy = DiseaseState('healthy', track_events=False, key='diarrhea')
    diarrhea = TransientDiseaseState('diarrhea')
    mild_diarrhea = DiseaseState('mild_diarrhea',
                                 disability_weight=get_disability_weight(
                                     healthstate_id=causes.mild_diarrhea.disability_weight),
                                 dwell_time=get_duration_in_days(causes.mild_diarrhea.duration))
    moderate_diarrhea = DiseaseState('moderate_diarrhea',
                                     disability_weight=get_disability_weight(
                                         healthstate_id=causes.moderate_diarrhea.disability_weight),
                                     dwell_time=get_duration_in_days(causes.moderate_diarrhea.duration))
    severe_diarrhea = ExcessMortalityState('severe_diarrhea',
                                           excess_mortality_data=get_severe_diarrhea_excess_mortality(),
                                           csmr_data=get_cause_specific_mortality(causes.severe_diarrhea.mortality),
                                           disability_weight=get_disability_weight(
                                               healthstate_id=causes.severe_diarrhea.disability_weight),
                                           dwell_time=get_duration_in_days(causes.severe_diarrhea.duration))

    diarrhea_transition = healthy.add_transition(diarrhea, triggered=True)
    healthy.allow_self_transitions()
    diarrhea.add_transition(mild_diarrhea, proportion=get_severity_splits(
            causes.diarrhea.incidence, causes.mild_diarrhea.incidence))
    diarrhea.add_transition(moderate_diarrhea, proportion=get_severity_splits(
            causes.diarrhea.incidence, causes.moderate_diarrhea.incidence))
    diarrhea.add_transition(severe_diarrhea, proportion=get_severity_splits(
            causes.diarrhea.incidence, causes.severe_diarrhea.incidence))
    mild_diarrhea.add_transition(healthy)
    moderate_diarrhea.add_transition(healthy)
    severe_diarrhea.add_transition(healthy)

    @uses_columns(['diarrhea'], 'alive')
    def cause_diarrhea(index, population_view):
        if not index.empty:
            diarrhea_transition.set_active(index)
            healthy.next_state(index, population_view)
            diarrhea_transition.set_inactive(index)

    etiology_names = ['{}'.format(name) for name, etiology in causes.items() if 'gbd_parent_cause' in etiology and
                      etiology.gbd_parent_cause == causes.diarrhea.gbd_cause]
    etiologies = [build_etiology_model(name, cause_diarrhea) for name in etiology_names]
    etiologies.append(build_etiology_model('unattributed', cause_diarrhea))

    def etiology_recovery_factory(etiology):
        @uses_columns([etiology.name], 'alive')
        def reset_etiology(index, population_view):
            if not index.empty:
                etiology.recovery_transition.set_active(index)
                etiology.pre_trigger_state.next_state(index, population_view)
                etiology.recovery_transition.set_inactive(index)

        return reset_etiology

    recovery_side_effects = [etiology_recovery_factory(etiology) for etiology in etiologies]

    def reset_etiologies(index):
        for side_effect in recovery_side_effects:
            side_effect(index)

    healthy.side_effect_function = reset_etiologies

    return [etiology.model for etiology in etiologies] + [DiseaseModel(
        'diarrhea', states=[healthy, diarrhea, mild_diarrhea, moderate_diarrhea, severe_diarrhea])]
