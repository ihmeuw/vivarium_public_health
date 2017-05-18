from collections import namedtuple

import pandas as pd

from ceam.framework.state_machine import TransitionSet
from ceam.framework.population import uses_columns

from ceam_inputs import (get_severity_splits, get_excess_mortality,
                         get_disability_weight, get_cause_specific_mortality,
                         get_incidence, get_pafs, get_remission)
from ceam_inputs.gbd_mapping import causes, risk_factors

from ceam_public_health.components.disease import (DiseaseModel, DiseaseState,
                                                   ExcessMortalityState, ProportionTransition)

Etiology = namedtuple('Etiology', ['name', 'model', 'recovery_trigger', 'pre_trigger_state'])


def build_etiology_model(etiology_name, infection_side_effect=None):
    diarrhea_incidence = get_incidence(modelable_entity_id=causes.diarrhea.incidence)
    etiology_paf = get_etiology_paf(etiology_name)
    # Multiply diarrhea incidence by etiology paf to get etiology incidence
    etiology_incidence = diarrhea_incidence.append(etiology_paf).groupby(
        ['age', 'sex_id', 'year_id'])[['draw_{}'.format(i) for i in range(1000)]].prod().reset_index()

    model = DiseaseModel(etiology_name)
    healthy = DiseaseState('healthy', track_events=False, key=etiology_name)
    sick = DiseaseState('infected',
                        disability_weight=0,
                        side_effect_function=infection_side_effect)

    healthy.add_transition(sick, rates=etiology_incidence)
    recovery_trigger = sick.add_transition(healthy, triggered=True)
    model.add_states([healthy, sick])

    return Etiology(name=etiology_name,
                    model=model,
                    recovery_trigger=recovery_trigger,
                    pre_trigger_state=sick)


def build_diarrhea_model():
    diarrhea_model = DiseaseModel('diarrhea')

    healthy = DiseaseState('healthy', track_events=False, key='diarrhea')
    diarrhea = DiseaseState('diarrhea')
    diarrhea_trigger = healthy.add_transition(diarrhea, triggered=True)

    @uses_columns(['diarrhea'])
    def cause_diarrhea(index, population_view):
        diarrhea_trigger(index)
        healthy.next_state(index, population_view)

    etiologies = [build_etiology_model(name, cause_diarrhea)
                  for name, etiology in risk_factors if causes.diarrhea in etiology.effected_causes]
    etiologies.append(build_etiology_model('unattributed', cause_diarrhea))

    @uses_columns(['{}'.format(name) for name, etiology in risk_factors if causes.diarrhea in etiology.effected_causes])
    def reset_etiologies(index, population_view):
        for disease in etiologies:
            disease.recovery_trigger(index)
            disease.pre_trigger_state.next_state(index, population_view)

    healthy.side_effect_function = reset_etiologies

    mild_diarrhea = DiseaseState('mild_diarrhea',
                                 disability_weight=get_disability_weight(
                                     causes.mild_diarrhea.disability_weight),
                                 dwell_time=get_duration_in_days(causes.mild_diarrhea.duration))
    moderate_diarrhea = DiseaseState('moderate_diarrhea',
                                     disability_weight=get_disability_weight(
                                         causes.moderate_diarrhea.disability_weight),
                                     dwell_time=get_duration_in_days(causes.moderate_diarrhea.duration))
    severe_diarrhea = ExcessMortalityState('severe_diarrhea',
                                           excess_mortality_data=get_severe_diarrhea_excess_mortality(),
                                           csmr_data=get_cause_specific_mortality(causes.severe_diarrhea.mortality),
                                           disability_weight=get_disability_weight(
                                               causes.severe_diarrhea.disability_weight),
                                           dwell_time=get_duration_in_days(causes.severe_diarrhea.duration))

    diarrhea_transitions = TransitionSet(
        ProportionTransition(mild_diarrhea, proportion=get_severity_splits(
            causes.diarrhea.incidence, causes.mild_diarrhea.incidence)),
        ProportionTransition(moderate_diarrhea, proportion=get_severity_splits(
            causes.diarrhea.incidence, causes.moderate_diarrhea.incidence)),
        ProportionTransition(severe_diarrhea, proportion=get_severity_splits(
            causes.diarrhea.incidence, causes.severe_diarrhea.incidence))
    )

    diarrhea.add_transition(diarrhea_transitions)
    mild_diarrhea.add_transition(healthy)
    moderate_diarrhea.add_transition(healthy)
    severe_diarrhea.add_transition(healthy)

    diarrhea_model.add_states([healthy, diarrhea, mild_diarrhea, moderate_diarrhea, severe_diarrhea])

    return diarrhea_model, [etiology.model for etiology in etiologies]


def get_severe_diarrhea_excess_mortality():
    diarrhea_excess_mortality = get_excess_mortality(causes.diarrhea.excess_mortality)
    severe_diarrhea_proportion = get_severity_splits(causes.diarrhea.incidence,
                                                     causes.severe_diarrhea.incidence)
    return diarrhea_excess_mortality.rate/severe_diarrhea_proportion


def get_duration_in_days(modelable_entity_id):
    """Get duration of disease for a modelable entity in days.

    Returns
    -------
    pandas.DataFrame
        Table with 'age', 'sex', 'year' and 'duration' columns
    """
    remission = get_remission(modelable_entity_id)
    duration = remission.copy()
    duration['duration'] = (1 / duration['remission']) *365
    duration.metadata = {'modelable_entity_id': modelable_entity_id}
    return duration[['year', 'age', 'duration', 'sex']]


def get_etiology_paf(etiology_name):
    if etiology_name == 'unattributed':
        attributable_etiologies = [name for name, etiology in risk_factors
                                   if causes.diarrhea in etiology.effected_causes]

        all_etiology_paf = pd.DataFrame()
        for etiology in attributable_etiologies:
            all_etiology_paf.append(get_etiology_paf(etiology))

        all_etiology_paf = all_etiology_paf.groupby(
            ['age', 'sex_id', 'year_id'])[['draw_{}'.format(i) for i in range(1000)]].sum()

        return pd.DataFrame(1 - all_etiology_paf,
                            columns=['draw_{}'.format(i) for i in range(1000)],
                            index=all_etiology_paf.index).reset_index()
    else:
        return get_pafs(risk_id=risk_factors[etiology_name].gbd_risk, cause_id=causes.diarrhea.gbd_cause)