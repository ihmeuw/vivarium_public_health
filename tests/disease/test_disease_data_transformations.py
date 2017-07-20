import numpy as np
import pandas as pd

from vivarium.test_util import build_table, get_randomness

from ceam_public_health.disease.data_transformations import (get_cause_level_prevalence, determine_if_sim_has_cause,
                                                             get_sequela_proportions,
                                                             determine_which_seq_diseased_sim_has)


def test_get_cause_level_prevalence():
    # pass in a states dict with only two sequela and make sure for one age/sex/year combo
    # that the value in cause_level_prevalence is equal to the sum of the two seq prevalences
    prev_df1 = build_table(0.03).rename(columns={'rate': 'prevalence'})[['year', 'age', 'prevalence', 'sex']]
    prev_df2 = build_table(0.02).rename(columns={'rate': 'prevalence'})[['year', 'age', 'prevalence', 'sex']]

    dict_of_disease_states = {'severe_heart_failure': prev_df1, 'moderate_heart_failure': prev_df2}
    cause_level, seq_level_dict = get_cause_level_prevalence(dict_of_disease_states, year_start=2005)

    # pick a random age and sex to test
    sex = "Male"
    age = 42

    # get a prevalence estimate for the random age and sex that we want to test
    moderate_heart_failure = seq_level_dict['moderate_heart_failure'].query(
        "age == {a} and sex == '{s}'".format(a=age, s=sex))
    seq_prevalence_1 = moderate_heart_failure['prevalence'].values[0]
    severe_heart_failure = seq_level_dict['severe_heart_failure'].query(
        "age == {a} and sex == '{s}'".format(a=age, s=sex))
    seq_prevalence_2 = severe_heart_failure['prevalence'].values[0]

    # add up the prevalences of the 2 sequela to see if we get cause-level prevalence
    cause_level = cause_level.query("age == {a} and sex == '{s}'".format(a=age, s=sex))
    cause_prev = cause_level['prevalence'].values[0]

    assert np.isclose(cause_prev, seq_prevalence_1 + seq_prevalence_2), ('get_cause_level_prevalence error. '
                                                                         'seq prevs need to add up to cause prev')
    assert np.allclose(cause_prev, .05), 'get_cause_level prevalence should match data from database as of 1/5/2017'


def test_determine_if_sim_has_cause():
    prevalence_df = pd.DataFrame({"age": [0, 5, 10, 15],
                                  "sex": ['Male']*4,
                                  "prevalence": [.25, .5, .75, 1],
                                  "year": [1990]*4})
    simulants_df = pd.DataFrame({'sex': ['Male']*500000,
                                 'age': [0, 5, 10, 15]*125000}, index=range(500000))
    results = determine_if_sim_has_cause(simulants_df, prevalence_df, get_randomness())
    grouped_results = results.groupby('age')[['condition_envelope']].sum()

    err_msg = "determine if sim has cause needs to appropriately assign causes based on prevalence"
    assert np.allclose(grouped_results.get_value(0, 'condition_envelope')/125000, .25, .01), err_msg
    assert np.allclose(grouped_results.get_value(5, 'condition_envelope')/125000, .5, .01), err_msg
    assert np.allclose(grouped_results.get_value(10, 'condition_envelope')/125000, .75, .01), err_msg
    assert np.allclose(grouped_results.get_value(15, 'condition_envelope')/125000, 1), err_msg


def test_get_sequela_proportions():
    cause_level_prevalence = pd.DataFrame({"age": [0, 5, 10, 15],
                                           "sex": ['Male']*4,
                                           "prevalence": [.25, .5, .75, 1],
                                           "year": 1990})

    seq_1_prevalence_df = cause_level_prevalence.copy()
    seq_2_prevalence_df = cause_level_prevalence.copy()
    seq_1_prevalence_df.prevalence = seq_1_prevalence_df['prevalence'] * .75
    seq_2_prevalence_df.prevalence = seq_2_prevalence_df['prevalence'] * .25
    states = dict({'sequela 1': seq_1_prevalence_df, 'sequela 2': seq_2_prevalence_df})

    df = get_sequela_proportions(cause_level_prevalence, states)

    assert list(df['sequela 1'].scaled_prevalence.values) == [.75]*4, "get_sequela_proportions"
    assert list(df['sequela 2'].scaled_prevalence.values) == [.25]*4, "get_sequela_proportions"


def test_determine_which_seq_diseased_sim_has():
    simulants_df = pd.DataFrame({'age': [0]*200000,
                                 'sex': ['Male']*200000,
                                 'condition_envelope': [False, True]*100000}, index=range(200000))

    df1 = pd.DataFrame({'age': [0, 10, 0, 10],
                        'sex': ['Male']*2 + ['Female']*2,
                        'scaled_prevalence': [.75, 1, .75, 1]})
    df2 = pd.DataFrame({'age': [0, 10, 0, 10],
                        'sex': ['Male']*2 + ['Female']*2,
                        'scaled_prevalence': [.25, 0, .25, 0]})
    sequela_proportion_dict = dict({'sequela 1': df1, 'sequela 2': df2})

    results = determine_which_seq_diseased_sim_has(sequela_proportion_dict, simulants_df, get_randomness())
    results['count'] = 1

    seq1 = results.query("condition_state == 'sequela 1'")
    seq2 = results.query("condition_state == 'sequela 2'")

    val1 = seq1.groupby('age')[['count']].sum()
    val1 = val1.get_value(0, 'count')
    val1 = val1 / 100000
    val2 = seq2.groupby('age')[['count']].sum()
    val2 = val2.get_value(0, 'count')
    val2 = val2 / 100000

    err_msg = "determine which seq diseased sim has needs to assign sequelae according to sequela prevalence"
    assert np.allclose(val1, .75, .1), err_msg
    assert np.allclose(val2, .25, .1), err_msg
