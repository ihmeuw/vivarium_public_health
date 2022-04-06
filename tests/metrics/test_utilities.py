from itertools import product

import numpy as np
import pandas as pd
import pytest
from vivarium.testing_utilities import metadata

from vivarium_public_health.metrics.utilities import (
    _MAX_AGE,
    _MIN_AGE,
    OutputTemplate,
    QueryString,
    get_age_bins,
    get_age_sex_filter_and_iterables,
    get_disease_event_counts,
    get_output_template,
    get_susceptible_person_time,
    get_years_lived_with_disability,
    to_years,
)


@pytest.fixture(params=((0, 100, 5, 1000), (20, 100, 5, 1000)))
def ages_and_bins(request):
    age_min = request.param[0]
    age_max = request.param[1]
    age_groups = request.param[2]
    num_ages = request.param[3]

    ages = np.linspace(age_min, age_max - age_groups / num_ages, num_ages)
    bin_ages, step = np.linspace(age_min, age_max, age_groups, endpoint=False, retstep=True)
    age_bins = pd.DataFrame(
        {
            "age_start": bin_ages,
            "age_end": bin_ages + step,
            "age_group_name": [str(name) for name in range(len(bin_ages))],
        }
    )

    return ages, age_bins


@pytest.fixture
def sexes():
    return ["Male", "Female"]


@pytest.fixture(params=list(product((True, False), repeat=3)))
def observer_config(request):
    c = {"by_age": request.param[0], "by_sex": request.param[1], "by_year": request.param[2]}
    return c


@pytest.fixture()
def builder(mocker):
    builder = mocker.MagicMock()
    df = pd.DataFrame(
        {
            "age_start": [0, 1, 4],
            "age_group_name": ["youngest", "younger", "young"],
            "age_end": [1, 4, 6],
        }
    )
    builder.data.load.return_value = df
    return builder


@pytest.mark.parametrize(
    "reference, test", product([QueryString(""), QueryString("abc")], [QueryString(""), ""])
)
def test_query_string_empty(reference, test):
    result = str(reference)
    assert reference + test == result
    assert reference + test == QueryString(result)
    assert isinstance(reference + test, QueryString)

    assert test + reference == result
    assert test + reference == QueryString(result)
    assert isinstance(test + reference, QueryString)

    reference += test
    assert reference == result
    assert reference == QueryString(result)
    assert isinstance(reference, QueryString)

    test += reference
    assert test == result
    assert test == QueryString(result)
    assert isinstance(test, QueryString)


@pytest.mark.parametrize("a, b", product([QueryString("a")], [QueryString("b"), "b"]))
def test_query_string(a, b):
    assert a + b == "a and b"
    assert a + b == QueryString("a and b")
    assert isinstance(a + b, QueryString)

    assert b + a == "b and a"
    assert b + a == QueryString("b and a")
    assert isinstance(b + a, QueryString)

    a += b
    assert a == "a and b"
    assert a == QueryString("a and b")
    assert isinstance(a, QueryString)

    b += a
    assert b == "b and a and b"
    assert b == QueryString("b and a and b")
    assert isinstance(b, QueryString)


def test_get_output_template(observer_config):
    template = get_output_template(**observer_config)

    assert isinstance(template, OutputTemplate)
    assert "${measure}" in template.template

    if observer_config["by_year"]:
        assert "_in_${year}" in template.template
    if observer_config["by_sex"]:
        assert "_among_${sex}" in template.template
    if observer_config["by_age"]:
        assert "_in_age_group_${age_group}" in template.template


@pytest.mark.parametrize(
    "measure, sex, age, year",
    product(
        ["test", "Test"], ["female", "Female"], [1.0, 1, "Early Neonatal"], [2011, "2011"]
    ),
)
def test_output_template(observer_config, measure, sex, age, year):
    template = get_output_template(**observer_config)

    out1 = template.substitute(measure=measure, sex=sex, age_group=age, year=year)
    out2 = (
        template.substitute(measure=measure)
        .substitute(sex=sex)
        .substitute(age_group=age)
        .substitute(year=year)
    )
    assert out1 == out2


def test_output_template_exact():
    template = get_output_template(by_age=True, by_sex=True, by_year=True)

    out = template.substitute(measure="Test", sex="Female", age_group=1.0, year=2011)
    expected = "test_in_2011_among_female_in_age_group_1.0"
    assert out == expected

    out = template.substitute(
        measure="Test", sex="Female", age_group="Early Neonatal", year=2011
    )
    expected = "test_in_2011_among_female_in_age_group_early_neonatal"

    assert out == expected


def test_get_age_sex_filter_and_iterables(ages_and_bins, observer_config):
    _, age_bins = ages_and_bins
    age_sex_filter, (ages, sexes) = get_age_sex_filter_and_iterables(
        observer_config, age_bins
    )

    assert isinstance(age_sex_filter, QueryString)
    if observer_config["by_age"] and observer_config["by_sex"]:
        assert age_sex_filter == '{age_start} <= age and age < {age_end} and sex == "{sex}"'

        for (g1, s1), (g2, s2) in zip(ages, age_bins.set_index("age_group_name").iterrows()):
            assert g1 == g2
            assert s1.equals(s2)

        assert sexes == ["Male", "Female"]

    elif observer_config["by_age"]:
        assert age_sex_filter == "{age_start} <= age and age < {age_end}"

        for (g1, s1), (g2, s2) in zip(ages, age_bins.set_index("age_group_name").iterrows()):
            assert g1 == g2
            assert s1.equals(s2)

        assert sexes == ["Both"]
    elif observer_config["by_sex"]:
        assert age_sex_filter == 'sex == "{sex}"'

        assert len(ages) == 1
        group, data = ages[0]
        assert group == "all_ages"
        assert data["age_start"] == _MIN_AGE
        assert data["age_end"] == _MAX_AGE

        assert sexes == ["Male", "Female"]

    else:
        assert age_sex_filter == ""

        assert len(ages) == 1
        group, data = ages[0]
        assert group == "all_ages"
        assert data["age_start"] == _MIN_AGE
        assert data["age_end"] == _MAX_AGE

        assert sexes == ["Both"]


def test_get_age_sex_filter_and_iterables_with_span(ages_and_bins, observer_config):
    _, age_bins = ages_and_bins
    age_sex_filter, (ages, sexes) = get_age_sex_filter_and_iterables(
        observer_config, age_bins, in_span=True
    )

    assert isinstance(age_sex_filter, QueryString)
    if observer_config["by_age"] and observer_config["by_sex"]:
        expected = '{age_start} < age_at_span_end and age_at_span_start < {age_end} and sex == "{sex}"'
        assert age_sex_filter == expected

        for (g1, s1), (g2, s2) in zip(ages, age_bins.set_index("age_group_name").iterrows()):
            assert g1 == g2
            assert s1.equals(s2)

        assert sexes == ["Male", "Female"]

    elif observer_config["by_age"]:
        assert (
            age_sex_filter
            == "{age_start} < age_at_span_end and age_at_span_start < {age_end}"
        )

        for (g1, s1), (g2, s2) in zip(ages, age_bins.set_index("age_group_name").iterrows()):
            assert g1 == g2
            assert s1.equals(s2)

        assert sexes == ["Both"]
    elif observer_config["by_sex"]:
        assert age_sex_filter == 'sex == "{sex}"'

        assert len(ages) == 1
        group, data = ages[0]
        assert group == "all_ages"
        assert data["age_start"] == _MIN_AGE
        assert data["age_end"] == _MAX_AGE

        assert sexes == ["Male", "Female"]

    else:
        assert age_sex_filter == ""

        assert len(ages) == 1
        group, data = ages[0]
        assert group == "all_ages"
        assert data["age_start"] == _MIN_AGE
        assert data["age_end"] == _MAX_AGE

        assert sexes == ["Both"]


def test_get_susceptible_person_time(ages_and_bins, sexes, observer_config):
    ages, age_bins = ages_and_bins
    disease = "test_disease"
    states = [f"susceptible_to_{disease}", disease]
    pop = pd.DataFrame(list(product(ages, sexes, states)), columns=["age", "sex", disease])
    pop["alive"] = "alive"
    # Shuffle the rows
    pop = pop.sample(frac=1).reset_index(drop=True)

    year = 2017
    step_size = pd.Timedelta(days=7)

    person_time = get_susceptible_person_time(
        pop, observer_config, disease, year, step_size, age_bins
    )

    values = set(person_time.values())
    assert len(values) == 1
    expected_value = to_years(step_size) * len(pop) / 2
    if observer_config["by_sex"]:
        expected_value /= 2
    if observer_config["by_age"]:
        expected_value /= len(age_bins)
    assert np.isclose(values.pop(), expected_value)

    # Doubling pop should double person time
    pop = pd.concat([pop, pop], axis=0, ignore_index=True)

    person_time = get_susceptible_person_time(
        pop, observer_config, disease, year, step_size, age_bins
    )

    values = set(person_time.values())
    assert len(values) == 1
    assert np.isclose(values.pop(), 2 * expected_value)


def test_get_disease_event_counts(ages_and_bins, sexes, observer_config):
    ages, age_bins = ages_and_bins
    disease = "test_disease"
    event_time = pd.Timestamp("1-1-2017")
    states = [event_time, pd.NaT]
    pop = pd.DataFrame(
        list(product(ages, sexes, states)), columns=["age", "sex", f"{disease}_event_time"]
    )
    # Shuffle the rows
    pop = pop.sample(frac=1).reset_index(drop=True)

    counts = get_disease_event_counts(pop, observer_config, disease, event_time, age_bins)

    values = set(counts.values())
    assert len(values) == 1
    expected_value = len(pop) / len(states)
    if observer_config["by_sex"]:
        expected_value /= 2
    if observer_config["by_age"]:
        expected_value /= len(age_bins)
    assert np.isclose(values.pop(), expected_value)

    # Doubling pop should double counts
    pop = pd.concat([pop, pop], axis=0, ignore_index=True)

    counts = get_disease_event_counts(pop, observer_config, disease, event_time, age_bins)

    values = set(counts.values())
    assert len(values) == 1
    assert np.isclose(values.pop(), 2 * expected_value)


def test_get_years_lived_with_disability(ages_and_bins, sexes, observer_config):
    alive = ["dead", "alive"]
    ages, age_bins = ages_and_bins
    causes = ["cause_a", "cause_b"]
    cause_a = ["susceptible_to_cause_a", "cause_a"]
    cause_b = ["susceptible_to_cause_b", "cause_b"]
    year = 2010
    step_size = pd.Timedelta(days=7)

    pop = pd.DataFrame(
        list(product(alive, ages, sexes, cause_a, cause_b)),
        columns=["alive", "age", "sex"] + causes,
    )
    # Shuffle the rows
    pop = pop.sample(frac=1).reset_index(drop=True)

    def disability_weight(cause):
        def inner(index):
            sub_pop = pop.loc[index]
            return pd.Series(1, index=index) * (sub_pop[cause] == cause)

        return inner

    disability_weights = {cause: disability_weight(cause) for cause in causes}

    ylds = get_years_lived_with_disability(
        pop, observer_config, year, step_size, age_bins, disability_weights, causes
    )

    values = set(ylds.values())
    assert len(values) == 1
    states_per_cause = len(cause_a)
    expected_value = len(pop) / (len(alive) * states_per_cause) * to_years(step_size)
    if observer_config["by_sex"]:
        expected_value /= 2
    if observer_config["by_age"]:
        expected_value /= len(age_bins)
    assert np.isclose(values.pop(), expected_value)

    # Doubling pop should double person time
    pop = pd.concat([pop, pop], axis=0, ignore_index=True)

    ylds = get_years_lived_with_disability(
        pop, observer_config, year, step_size, age_bins, disability_weights, causes
    )

    values = set(ylds.values())
    assert len(values) == 1
    assert np.isclose(values.pop(), 2 * expected_value)


@pytest.mark.parametrize(
    "age_start, exit_age, result_age_end_values, result_age_start_values",
    [
        (2, 5, {4, 5}, {2, 4}),
        (0, None, {1, 4, 6}, {0, 1, 4}),
        (1, 4, {4}, {1}),
        (1, 3, {3}, {1}),
        (0.8, 6, {1, 4, 6}, {0.8, 1, 4}),
    ],
)
def test_get_age_bins(
    builder, base_config, age_start, exit_age, result_age_end_values, result_age_start_values
):
    base_config.update(
        {"population": {"age_start": age_start, "exit_age": exit_age}}, **metadata(__file__)
    )
    builder.configuration = base_config
    df = get_age_bins(builder)
    assert set(df.age_end) == result_age_end_values
    assert set(df.age_start) == result_age_start_values
