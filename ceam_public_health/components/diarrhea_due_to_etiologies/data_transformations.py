import pandas as pd

from ceam_inputs import (get_excess_mortality, get_severity_splits,
                         get_remission, get_incidence, get_pafs, causes, risk_factors)


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
    etiology_incidence = etiology_incidence.drop('PAF', axis=1)
    return etiology_incidence