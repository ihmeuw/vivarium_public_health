import pandas as pd
import numpy as np

import pytest

from vivarium.testing_utilities import build_table
from vivarium.framework.configuration import build_simulation_configuration
from vivarium_public_health.risks import distributions

def test_should_rebin():
    test_config = build_simulation_configuration()
    test_config['population'] = {'population_size': 100}
    assert not distributions.should_rebin('test_risk', test_config)

    test_config['test_risk'] = {}
    assert not distributions.should_rebin('test_risk', test_config)

    test_config['test_risk'].rebin = False
    assert not distributions.should_rebin('test_risk', test_config)

    test_config['test_risk']['rebin'] = True
    assert distributions.should_rebin('test_risk', test_config)


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
        distributions.rebin_exposure_data(wrong_df)

    values = [0.1, 0.2, 0.3, 0.4]
    test_df = []
    for cat, value in zip(cats, values):
        test_df.append(build_table([cat, value], year_start, year_end, ('age','year', 'sex', 'parameter', 'value')))
    test_df = pd.concat(test_df)

    expected = []

    for cat, value in zip (['cat1', 'cat2'], [0.6, 0.4]):
        expected.append(build_table([cat, value], year_start, year_end, ('age', 'year', 'sex', 'parameter', 'value')))

    expected = pd.concat(expected).loc[:, ['age', 'year', 'sex', 'parameter', 'value']]
    rebinned = distributions.rebin_exposure_data(test_df).loc[:, expected.columns]
    expected = expected.set_index(['age', 'year','sex'])
    rebinned = rebinned.set_index(['age', 'year', 'sex'])

    assert np.allclose(expected.value[expected.parameter == 'cat1'], rebinned.value[rebinned.parameter=='cat1'])
    assert np.allclose(expected.value[expected.parameter == 'cat2'], rebinned.value[rebinned.parameter == 'cat2'])

