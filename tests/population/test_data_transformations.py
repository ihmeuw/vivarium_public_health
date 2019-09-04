import math

import numpy as np
import pandas as pd

from vivarium.testing_utilities import get_randomness, build_table
from vivarium_public_health.testing.utils import make_uniform_pop_data
import vivarium_public_health.population.data_transformations as dt


def test_assign_demographic_proportions():
    pop_data = dt.assign_demographic_proportions(make_uniform_pop_data(age_bin_midpoint=True))

    assert np.allclose(pop_data['P(sex, location, age| year)'], len(pop_data.year_start.unique()) / len(pop_data))
    assert np.allclose(
        pop_data['P(sex, location | age, year)'], (len(pop_data.year_start.unique())
                                                      * len(pop_data.age.unique()) / len(pop_data)))
    assert np.allclose(
        pop_data['P(age | year, sex, location)'], (len(pop_data.year_start.unique()) * len(pop_data.sex.unique())
                                                      * len(pop_data.location.unique()) / len(pop_data)))


def test_rescale_binned_proportions_full_range():
    pop_data = dt.assign_demographic_proportions(make_uniform_pop_data(age_bin_midpoint=True))
    pop_data = pop_data[pop_data.year_start == 1990]

    pop_data_scaled = dt.rescale_binned_proportions(pop_data, age_start=0, age_end=100)
    pop_data_scaled = pop_data_scaled[pop_data_scaled.age.isin(pop_data.age.unique())]

    assert np.allclose(pop_data['P(sex, location, age| year)'], pop_data_scaled['P(sex, location, age| year)'])


def test_rescale_binned_proportions_clipped_ends():
    pop_data = dt.assign_demographic_proportions(make_uniform_pop_data(age_bin_midpoint=True))
    pop_data = pop_data[pop_data.year_start == 1990]
    scale = len(pop_data.location.unique()) * len(pop_data.sex.unique())

    pop_data_scaled = dt.rescale_binned_proportions(pop_data, age_start=2, age_end=7)
    base_p = 1/len(pop_data)
    p_scaled = [base_p*7/5, base_p*3/5, base_p*2/5, base_p*8/5] + [base_p]*(len(pop_data_scaled)//scale - 5) + [0]

    for group, sub_population in pop_data_scaled.groupby(['sex', 'location']):
        assert np.allclose(sub_population['P(sex, location, age| year)'], p_scaled)


def test_rescale_binned_proportions_age_bin_edges():
    pop_data = dt.assign_demographic_proportions(make_uniform_pop_data(age_bin_midpoint=True))
    pop_data = pop_data[pop_data.year_start == 1990]

    # Test edge case where age_start/age_end fall on age bin boundaries.
    pop_data_scaled = dt.rescale_binned_proportions(pop_data, age_start=5, age_end=10)
    assert len(pop_data_scaled.age.unique()) == len(pop_data.age.unique()) + 2
    assert 7.5 in pop_data_scaled.age.unique()
    correct_data = ([1/len(pop_data)]*(len(pop_data_scaled)//2 - 2) + [0, 0])*2
    assert np.allclose(pop_data_scaled['P(sex, location, age| year)'], correct_data)


def test_smooth_ages():
    pop_data = dt.assign_demographic_proportions(make_uniform_pop_data(age_bin_midpoint=True))
    pop_data = pop_data[pop_data.year_start == 1990]
    simulants = pd.DataFrame({'age': [22.5]*10000 + [52.5]*10000,
                              'sex': ['Male', 'Female']*10000,
                              'location': [1, 2]*10000})
    randomness = get_randomness()
    smoothed_simulants = dt.smooth_ages(simulants, pop_data, randomness)

    assert math.isclose(len(smoothed_simulants.age.unique()), len(smoothed_simulants.index), abs_tol=1)
    # Tolerance is 3*std_dev of the sample mean
    assert math.isclose(smoothed_simulants.age.mean(), 37.5, abs_tol=3*math.sqrt(13.149778198**2/2000))


def test__get_bins_and_proportions_with_youngest_bin():
    pop_data = dt.assign_demographic_proportions(make_uniform_pop_data(age_bin_midpoint=True))
    pop_data = pop_data[(pop_data.year_start == 1990) & (pop_data.location == 1) & (pop_data.sex == 'Male')]
    age = dt.AgeValues(current=2.5, young=0, old=7.5)
    endpoints, proportions = dt._get_bins_and_proportions(pop_data, age)
    assert endpoints.left == 0
    assert endpoints.right == 5
    bin_width = endpoints.right - endpoints.left
    assert proportions.current == 1 / len(pop_data) / bin_width
    assert proportions.young == 1 / len(pop_data) / bin_width
    assert proportions.old == 1 / len(pop_data) / bin_width


def test__get_bins_and_proportions_with_oldest_bin():
    pop_data = dt.assign_demographic_proportions(make_uniform_pop_data(age_bin_midpoint=True))
    pop_data = pop_data[(pop_data.year_start == 1990) & (pop_data.location == 1) & (pop_data.sex == 'Male')]
    age = dt.AgeValues(current=97.5, young=92.5, old=100)
    endpoints, proportions = dt._get_bins_and_proportions(pop_data, age)
    assert endpoints.left == 95
    assert endpoints.right == 100
    bin_width = endpoints.right - endpoints.left
    assert proportions.current == 1 / len(pop_data) / bin_width
    assert proportions.young == 1 / len(pop_data) / bin_width
    assert proportions.old == 0


def test__get_bins_and_proportions_with_middle_bin():
    pop_data = dt.assign_demographic_proportions(make_uniform_pop_data(age_bin_midpoint=True))
    pop_data = pop_data[(pop_data.year_start == 1990) & (pop_data.location == 1) & (pop_data.sex == 'Male')]
    age = dt.AgeValues(current=22.5, young=17.5, old=27.5)
    endpoints, proportions = dt._get_bins_and_proportions(pop_data, age)
    assert endpoints.left == 20
    assert endpoints.right == 25
    bin_width = endpoints.right - endpoints.left
    assert proportions.current == 1 / len(pop_data) / bin_width
    assert proportions.young == 1 / len(pop_data) / bin_width
    assert proportions.old == 1 / len(pop_data) / bin_width


def test__construct_sampling_parameters():
    age = dt.AgeValues(current=50, young=22, old=104)
    endpoint = dt.EndpointValues(left=34, right=77)
    proportion = dt.AgeValues(current=0.1, young=0.5, old=0.3)

    pdf, slope, area, cdf_inflection_point = dt._construct_sampling_parameters(age, endpoint, proportion)

    assert pdf.left == ((proportion.current - proportion.young)/(age.current - age.young)
                        * (endpoint.left - age.young) + proportion.young)
    assert pdf.right == ((proportion.old - proportion.current) / (age.old - age.current)
                         * (endpoint.right - age.current) + proportion.current)
    assert area == 0.5 * ((proportion.current + pdf.left)*(age.current - endpoint.left)
                          + (pdf.right + proportion.current)*(endpoint.right - age.current))
    assert slope.left == (proportion.current - pdf.left) / (age.current - endpoint.left)
    assert slope.right == (pdf.right - proportion.current) / (endpoint.right - age.current)
    assert cdf_inflection_point == 1 / (2 * area) * (proportion.current + pdf.left) * (age.current - endpoint.left)


def test__compute_ages():
    assert dt._compute_ages(1, 10, 12, 0, 33) == 10 + 33/12*1
    assert dt._compute_ages(1, 10, 12, 5, 33) == 10 + 12/5*(np.sqrt(1+2*33*5/12**2*1) - 1)
