import pandas as pd
import numpy as np

import pytest

from vivarium.testing_utilities import build_table
from vivarium.framework.configuration import build_simulation_configuration
from vivarium_public_health.risks.data_transformation import *

def test_should_rebin():
    test_config = build_simulation_configuration()
    test_config['population'] = {'population_size': 100}
    assert not should_rebin('test_risk', test_config)

    test_config['test_risk'] = {}
    assert not should_rebin('test_risk', test_config)

    test_config['test_risk'].rebin = False
    assert not should_rebin('test_risk', test_config)

    test_config['test_risk']['rebin'] = True
    assert should_rebin('test_risk', test_config)


def test_rebin_exposure():
    cats = ['cat1', 'cat2', 'cat3', 'cat4']
    year_start = 2010
    year_end = 2013

    wrong_values = [0.1, 0.1, 0.1, 0.1]

    wrong_df = []
    for cat, value in zip(cats, wrong_values):
        wrong_df.append(build_table([cat, value], year_start, year_end, ('age','year', 'sex', 'parameter', 'value')))
    wrong_df = pd.concat(wrong_df)

    with pytest.raises(AssertionError):
        rebin_exposure_data(wrong_df)

    values = [0.1, 0.2, 0.3, 0.4]
    test_df = []
    for cat, value in zip(cats, values):
        test_df.append(build_table([cat, value], year_start, year_end, ('age','year', 'sex', 'parameter', 'value')))
    test_df = pd.concat(test_df)

    expected = []

    for cat, value in zip (['cat1', 'cat2'], [0.6, 0.4]):
        expected.append(build_table([cat, value], year_start, year_end, ('age', 'year', 'sex', 'parameter', 'value')))

    expected = pd.concat(expected).loc[:, ['age', 'year', 'sex', 'parameter', 'value']]
    rebinned = rebin_exposure_data(test_df).loc[:, expected.columns]
    expected = expected.set_index(['age', 'year','sex'])
    rebinned = rebinned.set_index(['age', 'year', 'sex'])

    assert np.allclose(expected.value[expected.parameter == 'cat1'], rebinned.value[rebinned.parameter=='cat1'])
    assert np.allclose(expected.value[expected.parameter == 'cat2'], rebinned.value[rebinned.parameter == 'cat2'])


def test_get_paf_data():
    cats = ['cat1', 'cat2', 'cat3', 'cat4']
    e_values = [0.1, 0.2, 0.3, 0.4]
    rr_values = [4, 3, 2, 1]
    year_start, year_end = 2000, 2010
    test_e = []
    for cat, value in zip(cats, e_values):
        test_e.append(build_table([cat, value],  year_start, year_end, ('age', 'year', 'sex', 'parameter', 'value')))
    test_e = pd.concat(test_e)

    test_rr = []
    for cat, value in zip(cats, rr_values):
        test_rr.append(build_table([cat, value], year_start, year_end, ('age', 'year', 'sex', 'parameter', 'value')))
    test_rr = pd.concat(test_rr)

    temp = sum(np.array(e_values) * np.array(rr_values))
    paf = (temp-1)/temp

    key_cols =['age', 'year', 'sex']
    paf_data = build_table(paf, year_start, year_end, ('age', 'year', 'sex', 'value'))[key_cols + ['value']].set_index(key_cols)
    get_paf = get_paf_data(test_e, test_rr)[key_cols +['value']].set_index(key_cols)

    assert np.allclose(paf_data.value, get_paf.value)
