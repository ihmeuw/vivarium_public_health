import pytest

from vivarium.testing_utilities import build_table
from vivarium_public_health.risks.data_transformation import *


def make_test_data_table(values, parameter='cat') -> pd.DataFrame:
    year_start = 1990  # same as the base config
    year_end = 2010

    if len(values) == 1:
        df = build_table(values[0], year_start, year_end, ('age', 'year', 'sex', 'value'))
    else:
        cats = [f'{parameter}{i+1}' for i in range(len(values))] if parameter == 'cat' else parameter
        df = []
        for cat, value in zip(cats, values):
            df.append(build_table([cat, value], year_start, year_end, ('age','year', 'sex', 'parameter', 'value')))
        df = pd.concat(df)
    return df


def test_should_rebin(base_config):
    assert not should_rebin('test_risk', base_config)

    base_config['test_risk'].rebin = False
    assert not should_rebin('test_risk', base_config)

    base_config['test_risk']['rebin'] = True
    assert should_rebin('test_risk', base_config)


@pytest.mark.parametrize('values', ([0.1, 0.1, 0.1, 0.1], [0.2, 0.3, 0.4]))
def test_rebin_exposure_fail(values):
    wrong_df = make_test_data_table(values)

    with pytest.raises(AssertionError):
        rebin_exposure_data(wrong_df)


@pytest.mark.parametrize(('initial', 'rebin'), [([0.1, 0.2, 0.3, 0.4], [0.6, 0.4]), ([0.2, 0.3, 0.3, 0.2], [0.8, 0.2])])
def test_rebin_exposure(initial, rebin):
    test_df = make_test_data_table(initial)
    expected = make_test_data_table(rebin)

    rebinned = rebin_exposure_data(test_df).loc[:, expected.columns]
    expected = expected.set_index(['age_group_start', 'year_start', 'sex'])
    rebinned = rebinned.set_index(['age_group_start', 'year_start', 'sex'])

    assert np.allclose(expected.value[expected.parameter == 'cat1'], rebinned.value[rebinned.parameter == 'cat1'])
    assert np.allclose(expected.value[expected.parameter == 'cat2'], rebinned.value[rebinned.parameter == 'cat2'])


test_data = [([0.3, 0.1, 0.1, 0.5], [3, 2.5, 2, 1], [(0.3 * 3 + 0.1 * 2.5 + 0.1 * 2) / (0.3+0.1+0.1), 1]),
             ([0.2, 0.5, 0.3], [5, 3, 1], [(0.2 * 5 + 0.5 * 3) / (0.2 + 0.5), 1])]
@pytest.mark.parametrize('e, rr, rebin', test_data)
def test_rebin_relative_risk(e, rr, rebin):
    exposure = make_test_data_table(e)
    relative_risk = make_test_data_table(rr)
    expected = make_test_data_table(rebin)

    rebinned = rebin_rr_data(relative_risk, exposure).loc[:, expected.columns]
    expected = expected.set_index(['age_group_start', 'year_start', 'sex'])
    rebinned = rebinned.set_index(['age_group_start', 'year_start', 'sex'])

    assert np.allclose(expected.value[expected.parameter == 'cat1'], rebinned.value[rebinned.parameter == 'cat1'])
    assert np.allclose(expected.value[expected.parameter == 'cat2'], rebinned.value[rebinned.parameter == 'cat2'])


test_data = [([0.1, 0.2, 0.3, 0.4], [4, 3, 2, 1], [0.5]), ([0.3, 0.6, 0.1], [20, 10, 1], [11.1/12.1])]
@pytest.mark.parametrize('e, rr, paf', test_data)
def test_get_paf_data(e, rr, paf):
    exposure = make_test_data_table(e)
    RR = make_test_data_table(rr)
    expected = make_test_data_table(paf)

    key_cols =['age_group_start', 'year_start', 'sex']
    get_paf = get_paf_data(exposure, RR)[key_cols + ['value']].set_index(key_cols)

    assert np.allclose(expected.value, get_paf.value)
