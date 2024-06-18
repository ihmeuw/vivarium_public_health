from itertools import product
from typing import Any, Dict, List

import hypothesis.strategies as st
import numpy as np
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
    if "age" not in parameter_columns:
        parameter_columns["age"] = (0, 125)
    return build_table(value, parameter_columns, key_columns, value_columns)


def make_uniform_pop_data(age_bin_midpoint=False):
    age_bins = [(b.age_start, b.age_end) for b in make_age_bins().itertuples()]
    sexes = ("Male", "Female")
    years = zip(range(1990, 2018), range(1991, 2019))
    locations = (1, 2)

    age_bins, sexes, years, locations = zip(*product(age_bins, sexes, years, locations))
    mins, maxes = zip(*age_bins)
    year_starts, year_ends = zip(*years)

    pop = pd.DataFrame(
        {
            "age_start": mins,
            "age_end": maxes,
            "sex": sexes,
            "year_start": year_starts,
            "year_end": year_ends,
            "location": locations,
            "value": 100 * (np.array(maxes) - np.array(mins)),
        }
    )
    if age_bin_midpoint:  # used for population tests
        pop["age"] = pop.apply(lambda row: (row["age_start"] + row["age_end"]) / 2, axis=1)
    return pop


def make_age_bins():
    idx = pd.MultiIndex.from_tuples(
        [
            (0.0, 0.01917808, "Early Neonatal"),
            (0.01917808, 0.07671233, "Late Neonatal"),
            (0.07671233, 1.0, "Post Neonatal"),
            (1.0, 5.0, "1 to 4"),
            (5.0, 10.0, "5 to 9"),
            (10.0, 15.0, "10 to 14"),
            (15.0, 20.0, "15 to 19"),
            (20.0, 25.0, "20 to 24"),
            (25.0, 30.0, "25 to 29"),
            (30.0, 35.0, "30 to 34"),
            (35.0, 40.0, "35 to 39"),
            (40.0, 45.0, "40 to 44"),
            (45.0, 50.0, "45 to 49"),
            (50.0, 55.0, "50 to 54"),
            (55.0, 60.0, "55 to 59"),
            (60.0, 65.0, "60 to 64"),
            (65.0, 70.0, "65 to 69"),
            (70.0, 75.0, "70 to 74"),
            (75.0, 80.0, "75 to 79"),
            (80.0, 85.0, "80 to 84"),
            (85.0, 90.0, "85 to 89"),
            (90.0, 95.0, "90 to 94"),
            (95.0, 125.0, "95 plus"),
        ],
        names=["age_start", "age_end", "age_group_name"],
    )
    return pd.DataFrame(index=idx).reset_index()
