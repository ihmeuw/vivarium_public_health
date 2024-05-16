import math

import numpy as np
import pandas as pd
import pytest
from vivarium.testing_utilities import get_randomness

import vivarium_public_health.population.data_transformations as dt
from vivarium_public_health.testing.utils import make_uniform_pop_data


@pytest.fixture
def pop_data(include_sex):
    pop_data = make_uniform_pop_data(age_bin_midpoint=True)
    # subset to the middle of the range where age bins are
    # the same size to make our math easier
    pop_data = pop_data.loc[(5 <= pop_data.age_start) & (pop_data.age_end <= 85)]
    pop_data = dt.assign_demographic_proportions(pop_data, include_sex=include_sex)
    return pop_data


@pytest.fixture
def pop_data_single_group():
    pop_data_single_group = make_uniform_pop_data(age_bin_midpoint=True)
    # subset to the first sex-location-age-year group
    g = pop_data_single_group.groupby(["sex", "location", "age", "year_start"])
    group = list(g.groups)[0]
    pop_data_single_group = g.get_group(group)
    sex = pop_data_single_group["sex"].iloc[0]
    pop_data_single_group = dt.assign_demographic_proportions(
        pop_data_single_group, include_sex=sex
    )
    return pop_data_single_group


def test_assign_demographic_proportions(pop_data, include_sex):
    male_scalar, female_scalar = {"Male": (2, 0), "Female": (0, 2), "Both": (1, 1)}[
        include_sex
    ]

    male_pop = pop_data[pop_data.sex == "Male"]
    female_pop = pop_data[pop_data.sex == "Female"]

    num_groups = len(pop_data)
    num_years = len(pop_data.year_start.unique())
    num_ages = len(pop_data.age.unique())
    num_sexes = len(pop_data.sex.unique())
    num_locations = len(pop_data.location.unique())

    for sex_scalar, subpop in ((male_scalar, male_pop), (female_scalar, female_pop)):
        assert np.allclose(
            subpop["P(sex, location, age| year)"],
            sex_scalar * num_years / num_groups,
        )
        assert np.allclose(
            subpop["P(sex, location | age, year)"],
            sex_scalar * num_ages * num_years / num_groups,
        )
        assert np.allclose(
            subpop["P(age | year, sex, location)"],
            num_years * num_sexes * num_locations / num_groups if sex_scalar else 0.0,
        )


def test_single_group_assign_demographic_proportions(pop_data_single_group):
    assert (pop_data_single_group["P(sex, location, age| year)"] == 1).all()
    assert (pop_data_single_group["P(sex, location | age, year)"] == 1).all()
    assert (pop_data_single_group["P(age | year, sex, location)"] == 1).all()


def test_rescale_binned_proportions_full_range(pop_data):
    pop_data = pop_data[pop_data.year_start == 1990].reset_index(drop=True)

    pop_data_scaled = dt.rescale_binned_proportions(pop_data, age_start=0, age_end=100)
    pop_data_scaled = pop_data_scaled[pop_data_scaled.age.isin(pop_data.age.unique())]

    assert np.allclose(
        pop_data["P(sex, location, age| year)"],
        pop_data_scaled["P(sex, location, age| year)"],
    )


def test_rescale_binned_proportions_clipped_ends(pop_data, include_sex):
    pop_data = pop_data[pop_data.year_start == 1990]
    scale = len(pop_data.location.unique()) * len(pop_data.sex.unique())

    pop_data_scaled = dt.rescale_binned_proportions(pop_data, age_start=7, age_end=12)
    base_p = 1 / len(pop_data)
    p_scaled = np.array(
        [base_p * 7 / 5, base_p * 3 / 5, base_p * 2 / 5, base_p * 8 / 5]
        + [base_p] * (len(pop_data_scaled) // scale - 5)
        + [0]
    )
    male_scalar, female_scalar = {
        "Male": (2, 0),
        "Female": (0, 2),
        "Both": (1, 1),
    }[include_sex]

    for group, sub_population in pop_data_scaled.groupby(["sex", "location"]):
        scalar = {"Male": male_scalar, "Female": female_scalar}[group[0]]
        assert np.allclose(sub_population["P(sex, location, age| year)"], scalar * p_scaled)


def test_rescale_binned_proportions_age_bin_edges(pop_data, include_sex):
    pop_data = pop_data[pop_data.year_start == 1990]

    # Test edge case where age_start/age_end fall on age bin boundaries.
    pop_data_scaled = dt.rescale_binned_proportions(pop_data, age_start=5, age_end=10)
    assert len(pop_data_scaled.age.unique()) == len(pop_data.age.unique()) + 2
    assert 7.5 in pop_data_scaled.age.unique()
    correct_data = np.zeros(len(pop_data_scaled))
    valid_ages = pop_data_scaled.age_end <= pop_data.age_end.max()
    correct_data[valid_ages] = 1 / len(pop_data)
    if include_sex in ["Male", "Female"]:
        correct_data[pop_data_scaled.sex == include_sex] *= 2
        correct_data[pop_data_scaled.sex != include_sex] *= 0
    assert np.allclose(pop_data_scaled["P(sex, location, age| year)"], correct_data)


def test_smooth_ages(pop_data, include_sex):
    pop_data = pop_data[pop_data.year_start == 1990]

    simulants = pd.DataFrame(
        {
            "age": [22.5] * 10000 + [52.5] * 10000,
            "sex": (
                ["Male", "Female"] * 10000 if include_sex == "Both" else [include_sex] * 20000
            ),
            "location": [1, 2] * 10000,
        }
    )
    randomness = get_randomness()
    smoothed_simulants = dt.smooth_ages(simulants, pop_data, randomness)

    assert math.isclose(
        len(smoothed_simulants.age.unique()), len(smoothed_simulants.index), abs_tol=1
    )
    # Tolerance is 3*std_dev of the sample mean
    assert math.isclose(
        smoothed_simulants.age.mean(), 37.5, abs_tol=3 * math.sqrt(13.149778198**2 / 2000)
    )


def test__get_bins_and_proportions_with_youngest_bin():
    pop_data = dt.assign_demographic_proportions(
        make_uniform_pop_data(age_bin_midpoint=True),
        include_sex="Male",
    )
    pop_data = pop_data[
        (pop_data.year_start == 1990) & (pop_data.location == 1) & (pop_data.sex == "Male")
    ].sort_values("age")
    age = dt.AgeValues(current=pop_data["age"].iloc[0], young=0, old=pop_data["age"].iloc[1])
    endpoints, proportions = dt._get_bins_and_proportions(pop_data, age)
    assert endpoints.left == 0
    assert endpoints.right == pop_data["age_end"].iloc[0]
    bin_width = endpoints.right - endpoints.left
    expected_proportion = pop_data["P(age | year, sex, location)"].iloc[0] / bin_width
    for p in proportions:
        assert math.isclose(p, expected_proportion)


def test__get_bins_and_proportions_with_oldest_bin():
    pop_data = dt.assign_demographic_proportions(
        make_uniform_pop_data(age_bin_midpoint=True),
        include_sex="Male",
    )
    pop_data = pop_data[
        (pop_data.year_start == 1990) & (pop_data.location == 1) & (pop_data.sex == "Male")
    ].sort_values("age")
    age = dt.AgeValues(
        current=pop_data["age"].iloc[-1],
        young=pop_data["age"].iloc[-2],
        old=pop_data["age_end"].iloc[-1],
    )
    endpoints, proportions = dt._get_bins_and_proportions(pop_data, age)
    assert endpoints.left == pop_data["age_start"].iloc[-1]
    assert endpoints.right == pop_data["age_end"].iloc[-1]
    bin_width = endpoints.right - endpoints.left
    expected_proportion = pop_data["P(age | year, sex, location)"].iloc[-1] / bin_width
    assert proportions.current == expected_proportion
    assert proportions.young == expected_proportion
    assert proportions.old == 0


def test__get_bins_and_proportions_with_middle_bin():
    pop_data = dt.assign_demographic_proportions(
        make_uniform_pop_data(age_bin_midpoint=True),
        include_sex="Male",
    )
    pop_data = pop_data[
        (pop_data.year_start == 1990) & (pop_data.location == 1) & (pop_data.sex == "Male")
    ]
    age = dt.AgeValues(current=22.5, young=17.5, old=27.5)
    endpoints, proportions = dt._get_bins_and_proportions(pop_data, age)
    assert endpoints.left == 20
    assert endpoints.right == 25
    bin_width = endpoints.right - endpoints.left
    expected_proportion = (
        pop_data.loc[pop_data.age == age.current, "P(age | year, sex, location)"].iloc[0]
    ) / bin_width
    for p in proportions:
        assert math.isclose(p, expected_proportion)


def test__construct_sampling_parameters():
    age = dt.AgeValues(current=50, young=22, old=104)
    endpoint = dt.EndpointValues(left=34, right=77)
    proportion = dt.AgeValues(current=0.1, young=0.5, old=0.3)

    pdf, slope, area, cdf_inflection_point = dt._construct_sampling_parameters(
        age, endpoint, proportion
    )

    assert pdf.left == (
        (proportion.current - proportion.young)
        / (age.current - age.young)
        * (endpoint.left - age.young)
        + proportion.young
    )
    assert pdf.right == (
        (proportion.old - proportion.current)
        / (age.old - age.current)
        * (endpoint.right - age.current)
        + proportion.current
    )
    assert area == 0.5 * (
        (proportion.current + pdf.left) * (age.current - endpoint.left)
        + (pdf.right + proportion.current) * (endpoint.right - age.current)
    )
    assert slope.left == (proportion.current - pdf.left) / (age.current - endpoint.left)
    assert slope.right == (pdf.right - proportion.current) / (endpoint.right - age.current)
    assert cdf_inflection_point == 1 / (2 * area) * (proportion.current + pdf.left) * (
        age.current - endpoint.left
    )


def test__compute_ages():
    assert dt._compute_ages(1, 10, 12, 0, 33) == 10 + 33 / 12 * 1
    assert dt._compute_ages(1, 10, 12, 5, 33) == 10 + 12 / 5 * (
        np.sqrt(1 + 2 * 33 * 5 / 12**2 * 1) - 1
    )
