from collections import namedtuple

import pandas as pd

from ceam.framework.population import uses_columns

from ceam_inputs import (get_severity_splits, get_excess_mortality,
                         get_disability_weight, get_cause_specific_mortality,
                         get_incidence, get_pafs, get_remission)
from ceam_inputs.gbd_mapping import causes, risk_factors

from ceam_public_health.components.disease import (DiseaseModel, DiseaseState,
                                                   TransientDiseaseState, ExcessMortalityState)

Etiology = namedtuple('Etiology', ['name', 'model', 'recovery_transition', 'pre_trigger_state'])


def build_etiology_model(etiology_name, infection_side_effect=None):
    etiology_incidence = get_etiology_incidence(etiology_name)
    model = DiseaseModel(etiology_name)
    healthy = DiseaseState('healthy', track_events=False, key=etiology_name)
    sick = DiseaseState(etiology_name,
                        disability_weight=0,
                        side_effect_function=infection_side_effect)

    healthy.add_transition(sick, rates=etiology_incidence)
    healthy.allow_self_transitions()
    recovery_transition = sick.add_transition(healthy, triggered=True)
    model.add_states([healthy, sick])

    return Etiology(name=etiology_name,
                    model=model,
                    recovery_transition=recovery_transition,
                    pre_trigger_state=sick)


def build_diarrhea_model():
    diarrhea_model = DiseaseModel('diarrhea')

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
        diarrhea_transition.set_active(index)
        healthy.next_state(index, population_view)
        diarrhea_transition.set_inactive(index)

    etiology_names = ['{}'.format(name) for name, etiology in risk_factors.items()
                      if causes.diarrhea in etiology.effected_causes]
    etiologies = [build_etiology_model(name, cause_diarrhea) for name in etiology_names]
    etiologies.append(build_etiology_model('unattributed', cause_diarrhea))

    @uses_columns(etiology_names, 'alive')
    def reset_etiologies(index, population_view):
        for disease in etiologies:
            disease.recovery_transition.set_active(index)
            disease.pre_trigger_state.next_state(index, population_view)
            disease.recovery_transition.set_inactive(index)

    healthy.side_effect_function = reset_etiologies

    diarrhea_model.add_states([healthy, diarrhea, mild_diarrhea, moderate_diarrhea, severe_diarrhea])

    return [etiology.model for etiology in etiologies] + [diarrhea_model]


def get_severe_diarrhea_excess_mortality():
    diarrhea_excess_mortality = get_excess_mortality(causes.diarrhea.excess_mortality)
    severe_diarrhea_proportion = get_severity_splits(causes.diarrhea.incidence,
                                                     causes.severe_diarrhea.incidence)
    diarrhea_excess_mortality['rate'] = diarrhea_excess_mortality['rate']/severe_diarrhea_proportion
    return diarrhea_excess_mortality


def get_duration_in_days(modelable_entity_id):
    """Get duration of disease for a modelable entity in days.

    Returns
    -------
    pandas.DataFrame
        Table with 'age', 'sex', 'year' and 'duration' columns
    """
    remission = get_remission(modelable_entity_id)
    duration = remission.copy()
    duration['duration'] = (1 / duration['remission']) * 365
    duration.metadata = {'modelable_entity_id': modelable_entity_id}
    return duration[['year', 'age', 'duration', 'sex']]


def get_etiology_paf(etiology_name):
    if etiology_name == 'unattributed':
        attributable_etiologies = [name for name, etiology in risk_factors.items()
                                   if causes.diarrhea in etiology.effected_causes]

        all_etiology_paf = pd.concat([get_etiology_paf(name) for name in attributable_etiologies])

        grouped = all_etiology_paf.groupby(['age', 'sex', 'year']).sum()
        pafs = grouped[['PAF']].values

        pafs = pd.DataFrame(1 - pafs,
                            columns=['PAF'],
                            index=grouped.index).reset_index()
    else:
        pafs = get_pafs(risk_id=risk_factors[etiology_name].gbd_risk, cause_id=causes.diarrhea.gbd_cause)

    draws = pafs._get_numeric_data()
    draws[draws < 0] = 0

    return pafs


def get_etiology_incidence(etiology_name):
    diarrhea_incidence = get_incidence(modelable_entity_id=causes.diarrhea.incidence)
    etiology_paf = get_etiology_paf(etiology_name)
    # Multiply diarrhea incidence by etiology paf to get etiology incidence
    etiology_incidence = pd.merge(diarrhea_incidence, etiology_paf, on=['age', 'sex', 'year'])
    etiology_incidence['rate'] = etiology_incidence['rate'] * etiology_incidence['PAF']
    #print(etiology_incidence)
    etiology_incidence = etiology_incidence.drop('PAF', axis=1)
    return etiology_incidence

