from hypothesis import given
import hypothesis.strategies as st
import pytest

from vivarium_public_health.risks.data_transformations import RiskString, TargetString


@st.composite
def test_string(draw, min_components=0, max_components=None):
    alphabet = st.characters(blacklist_characters=['.'])
    string_parts = draw(st.lists(st.text(alphabet=alphabet), min_size=min_components, max_size=max_components))
    return '.'.join(string_parts)


@given(test_string().filter(lambda x: len(x.split('.')) != 2))
def test_RiskString_fail(s):
    with pytest.raises(ValueError):
        RiskString(s)

@pytest.mark.parametrize('string', ['a.b.c', 'a'])
def test_RiskString_fail(string):
    with pytest.raises(ValueError):
        RiskString(string)


@pytest.mark.parametrize('string', ['risk_factor.risk', 'coverage_gap.cg', 'alternative_risk_factor.risk'])
def test_RiskString_pass(string):
    risk_type, risk_name = string.split('.')
    r = RiskString(string)
    assert r.type == risk_type
    assert r.name == risk_name


@pytest.mark.parametrize('string', ['a', 'a.b', 'a.b.c.d'])
def test_TargetString_fail(string):
    with pytest.raises(ValueError):
        TargetString(string)


@pytest.mark.parametrize('string', ['risk_factor.risk.exposure_parameters',
                                    'cause.disease.incidence_rate',
                                    'cause.disease.excess_mortality'])
def test_RiskString_pass(string):
    target_type, target_name, target_measure = string.split('.')
    t = TargetString(string)
    assert t.type == target_type
    assert t.name == target_name
    assert t.measure == target_measure





