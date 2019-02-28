from hypothesis import given
import hypothesis.strategies as st
import hypothesis.extra.pandas as pdst
import pytest
import pandas as pd

from vivarium_public_health.risks.data_transformations import (RiskString, TargetString,
                                                               _rebin_exposure_data, _rebin_relative_risk_data)


@st.composite
def component_string(draw, min_components=0, max_components=None):
    alphabet = st.characters(blacklist_characters=['.'])
    string_parts = draw(st.lists(st.text(alphabet=alphabet), min_size=min_components, max_size=max_components))
    return '.'.join(string_parts)


@given(component_string().filter(lambda x: len(x.split('.')) != 2))
def test_RiskString_fail(s):
    with pytest.raises(ValueError):
        RiskString(s)


@given(component_string(2, 2))
def test_RiskString_pass(s):
    risk_type, risk_name = s.split('.')
    r = RiskString(s)
    assert r.type == risk_type
    assert r.name == risk_name


@given(component_string().filter(lambda x: len(x.split('.')) != 3))
def test_TargetString_fail(s):
    with pytest.raises(ValueError):
        TargetString(s)


@given(component_string(3, 3))
def test_TargetString_pass(s):
    target_type, target_name, target_measure = s.split('.')
    t = TargetString(s)
    assert t.type == target_type
    assert t.name == target_name
    assert t.measure == target_measure


@pytest.mark.parametrize('rebin_categories, rebinned_values', [({'cat1', 'cat2'}, (0.7, 0.3)),
                                                               ({'cat1'}, (0.5, 0.5)),
                                                               ({'cat2'}, (0.2, 0.8)),
                                                               ({'cat2', 'cat3'}, (0.5, 0.5)),
                                                               ({'cat1', 'cat3'}, (0.8, 0.2))])
def test__rebin_exposure_data(rebin_categories, rebinned_values):
    df = pd.DataFrame({'year': [1990, 1990, 1995, 1995]*3,
                       'age': [10, 40, 10, 40]*3,
                       'parameter': ['cat1']*4 + ['cat2']*4 + ['cat3']*4,
                       'value': [0.5]*4 + [0.2]*4 + [0.3]*4})
    rebinned_df = _rebin_exposure_data(df, rebin_categories)

    assert rebinned_df.shape == (8, 4)
    assert (rebinned_df[rebinned_df.parameter == 'cat1'].value == rebinned_values[0]).all()
    assert (rebinned_df[rebinned_df.parameter == 'cat2'].value == rebinned_values[1]).all()


@pytest.mark.parametrize('rebin_categories, rebinned_values', [({'cat1', 'cat2'}, (10, 1)),
                                                               ({'cat1'}, (0, 7.3)),
                                                               ({'cat2'}, (10, 1)),
                                                               ({'cat2', 'cat3'}, (7.3, 0)),
                                                               ({'cat1', 'cat3'}, (1, 10))])
def test__rebin_relative_risk(rebin_categories, rebinned_values):
    exp = pd.DataFrame({'year': [1990, 1990, 1995, 1995]*3,
                        'age': [10, 40, 10, 40]*3,
                        'parameter': ['cat1']*4 + ['cat2']*4 + ['cat3']*4,
                        'value': [0.0]*4 + [0.7]*4 + [0.3]*4})

    rr = pd.DataFrame({'year': [1990, 1990, 1995, 1995]*3,
                       'age': [10, 40, 10, 40]*3,
                       'parameter': ['cat1']*4 + ['cat2']*4 + ['cat3']*4,
                       'value': [5]*4 + [10]*4 + [1]*4})

    rebinned_df = _rebin_relative_risk_data(rr, exp, rebin_categories)

    assert rebinned_df.shape == (8, 4)
    assert (rebinned_df[rebinned_df.parameter == 'cat1'].value == rebinned_values[0]).all()
    assert (rebinned_df[rebinned_df.parameter == 'cat2'].value == rebinned_values[1]).all()
