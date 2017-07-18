from itertools import product

import numpy as np
import pandas as pd

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


def test_rescale_binned_proportions():
    pop_data = dt.assign_demographic_proportions(make_uniform_pop_data())
    pop_data = pop_data[pop_data.year == 1990]

    pop_data_scaled = dt.rescale_binned_proportions(pop_data, pop_age_start=0, pop_age_end=100)
    # Should be a no-op
    assert np.allclose(pop_data['P(sex, location_id, age| year)'], pop_data_scaled['P(sex, location_id, age| year)'])

    pop_data_scaled = dt.rescale_binned_proportions(pop_data, pop_age_start=2.5, pop_age_end=7.5)
    base_p = 1/len(pop_data)
    p_scaled = [base_p/2, base_p, base_p/2]

    for sex, location_id in product(['Male', 'Female'], pop_data_scaled.location_id.unique()):
        sub_pop_scaled = pop_data_scaled[(pop_data_scaled.sex == sex) & (pop_data_scaled.location_id == location_id)]
        assert np.allclose(sub_pop_scaled['P(sex, location_id, age| year)'])


def test_smooth_ages():
    pass


def test__get_bins_and_proportions():
    pop_data = dt.assign_demographic_proportions(make_uniform_pop_data())
    pop_data = pop_data[(pop_data.year == 1990) & (pop_data.location_id == 1) & (pop_data.sex == 'Male')]
    age = dt.AgeValues(current=2.5, young=0, old=5)
    endpoints, proportions = dt._get_bins_and_proportions(pop_data, age)
    assert endpoints.left == 0
    assert endpoints.right == 5
    assert proportions.current ==




def test__construct_sampling_parameters():



def test__compute_ages():
    assert dt._compute_ages(1, 10, 12, 0, 33) == 10 + 33/12*1
    assert dt._compute_ages(1, 10, 12, 5, 33) == 10 + 12/5*(np.sqrt(1+2*33*5/12**2*1) - 1)
