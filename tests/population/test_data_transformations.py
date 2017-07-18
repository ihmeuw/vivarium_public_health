from itertools import product
import math

import numpy as np
import pandas as pd

from vivarium.test_util import get_randomness

import ceam_public_health.population.data_transformations as dt


def make_uniform_pop_data():
    age_bins = [(n, n+2.5, n+5) for n in range(0, 100, 5)]
    sexes = ('Male', 'Female', 'Both')
    years = (1990, 1995, 2000, 2005)
    locations = (1, 2)

    age_bins, sexes, years, locations = zip(*product(age_bins, sexes, years, locations))
    mins, ages, maxes = zip(*age_bins)
    pop = pd.DataFrame({'age': ages,
                        'age_group_start': mins,
                        'age_group_end': maxes,
                        'sex': sexes,
                        'year': years,
                        'location_id': locations,
                        'pop_scaled': [100]*len(ages)})
    pop.loc[pop.sex == 'Both', 'pop_scaled'] = 200
    return pop


def test_assign_demographic_proportions():
    pop_data = dt.assign_demographic_proportions(make_uniform_pop_data())

    assert not pop_data[pop_data.sex == 'Both'].any()
    # The make_uniform_pop_data has four years by default, so we should have four uniform conditional pdfs
    assert np.allclose(pop_data['P(sex, location_id, age| year)'], 4 / len(pop_data))
    # There is a single location and the sexes are uniformly distributed.
    assert np.allclose(pop_data['P(sex, location_id | age, year)'], 1/2)


def test_rescale_binned_proportions_full_range():
    pop_data = dt.assign_demographic_proportions(make_uniform_pop_data())
    pop_data = pop_data[pop_data.year == 1990]

    pop_data_scaled = dt.rescale_binned_proportions(pop_data, pop_age_start=0, pop_age_end=100)
    # Should be a no-op
    assert np.allclose(pop_data['P(sex, location_id, age| year)'], pop_data_scaled['P(sex, location_id, age| year)'])


def test_rescale_binned_proportions_clipped_ends():
    pop_data = dt.assign_demographic_proportions(make_uniform_pop_data())
    pop_data = pop_data[pop_data.year == 1990]

    pop_data_scaled = dt.rescale_binned_proportions(pop_data, pop_age_start=2.5, pop_age_end=7.5)
    base_p = 1/len(pop_data)
    p_scaled = [base_p/2, base_p, base_p/2]

    for sex, location_id in product(['Male', 'Female'], pop_data_scaled.location_id.unique()):
        sub_pop_scaled = pop_data_scaled[(pop_data_scaled.sex == sex) & (pop_data_scaled.location_id == location_id)]
        assert np.allclose(sub_pop_scaled['P(sex, location_id, age| year)'], p_scaled)


def test_rescale_binned_proportions_age_bin_edges():
    pop_data = dt.assign_demographic_proportions(make_uniform_pop_data())
    pop_data = pop_data[pop_data.year == 1990]

    # Test edge case where pop_age_start/pop_age_end fall on age bin boundaries.
    pop_data_scaled = dt.rescale_binned_proportions(pop_data, pop_age_start=5, pop_age_end=10)
    assert len(pop_data_scaled.age.unique()) == 1
    assert 7.5 in pop_data_scaled.age.unique()
    assert np.allclose(pop_data_scaled['P(sex, location_id, age| year)'], base_p)



def test_smooth_ages():
    pop_data = make_uniform_pop_data()
    simulants = pd.DataFrame({'age': [22.5]*1000 + [52.5]*1000,
                              'sex': ['Male', 'Female']*1000,
                              'location': [1, 2]*1000})
    randomness = get_randomness()
    smoothed_simulants = dt.smooth_ages(simulants, pop_data, randomness)

    assert math.isclose(len(smoothed_simulants.ages.unique()), len(smoothed_simulants.index), abs_tol=1)
    # Tolerance is 3*std_dev of the sample mean
    assert math.isclose(smoothed_simulants.ages.mean(), 42.5, abs_tol=3*math.sqrt(13.149778198**2/2000))


def test__get_bins_and_proportions_with_youngest_bin():
    pop_data = dt.assign_demographic_proportions(make_uniform_pop_data())
    pop_data = pop_data[(pop_data.year == 1990) & (pop_data.location_id == 1) & (pop_data.sex == 'Male')]
    age = dt.AgeValues(current=2.5, young=0, old=7.5)
    endpoints, proportions = dt._get_bins_and_proportions(pop_data, age)
    assert endpoints.left == 0
    assert endpoints.right == 5
    assert proportions.current == 1 / len(pop_data)
    assert proportions.left == 1 / len(pop_data)
    assert proportions.right == 1 / len(pop_data)


def test__get_bins_and_proportions_with_oldest_bin():
    pop_data = dt.assign_demographic_proportions(make_uniform_pop_data())
    pop_data = pop_data[(pop_data.year == 1990) & (pop_data.location_id == 1) & (pop_data.sex == 'Male')]
    age = dt.AgeValues(current=97.5, young=92.5, old=100)
    endpoints, proportions = dt._get_bins_and_proportions(pop_data, age)
    assert endpoints.left == 95
    assert endpoints.right == 100
    assert proportions.current == 1 / len(pop_data)
    assert proportions.left == 1 / len(pop_data)
    assert proportions.right == 0


def test__get_bins_and_proportions_with_middle_bin():
    pop_data = dt.assign_demographic_proportions(make_uniform_pop_data())
    pop_data = pop_data[(pop_data.year == 1990) & (pop_data.location_id == 1) & (pop_data.sex == 'Male')]
    age = dt.AgeValues(current=22.5, young=17.5, old=27.5)
    endpoints, proportions = dt._get_bins_and_proportions(pop_data, age)
    assert endpoints.left == 20
    assert endpoints.right == 25
    assert proportions.current == 1 / len(pop_data)
    assert proportions.left == 1 / len(pop_data)
    assert proportions.right == 1 / len(pop_data)


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
    assert cdf_inflection_point == 1 / (2 * area) * (proportion.age + pdf.left) * (age.current - endpoint.left)


def test__compute_ages():
    assert dt._compute_ages(1, 10, 12, 0, 33) == 10 + 33/12*1
    assert dt._compute_ages(1, 10, 12, 5, 33) == 10 + 12/5*(np.sqrt(1+2*33*5/12**2*1) - 1)
