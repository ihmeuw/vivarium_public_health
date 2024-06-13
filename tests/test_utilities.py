from typing import Any, Dict, List

import hypothesis.strategies as st
import pandas as pd
import pytest
from hypothesis import given
from vivarium.testing_utilities import build_table

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


def build_table_with_age(
    value: Any,
    parameter_columns: Dict = {"year": (1990, 2020)},
    key_columns: Dict = {"sex": ("Female", "Male")},
    value_columns: List = ["value"],
) -> pd.DataFrame:
    parameter_columns["age"] = (0, 125)
    return build_table(value, parameter_columns, key_columns, value_columns)
