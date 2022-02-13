import hypothesis.strategies as st
import pytest
from hypothesis import given

from vivarium_public_health.utilities import EntityString, TargetString


@st.composite
def component_string(draw, min_components=0, max_components=None):
    alphabet = st.characters(blacklist_characters=["."])
    string_parts = draw(
        st.lists(st.text(alphabet=alphabet), min_size=min_components, max_size=max_components)
    )
    return ".".join(string_parts)


@given(component_string().filter(lambda x: len(x.split(".")) != 2))
def test_EntityString_fail(s):
    with pytest.raises(ValueError):
        EntityString(s)


@given(component_string(2, 2))
def test_EntityString_pass(s):
    entity_type, entity_name = s.split(".")
    r = EntityString(s)
    assert r.type == entity_type
    assert r.name == entity_name


@given(component_string().filter(lambda x: len(x.split(".")) != 3))
def test_TargetString_fail(s):
    with pytest.raises(ValueError):
        TargetString(s)


@given(component_string(3, 3))
def test_TargetString_pass(s):
    target_type, target_name, target_measure = s.split(".")
    t = TargetString(s)
    assert t.type == target_type
    assert t.name == target_name
    assert t.measure == target_measure
