from itertools import product

import numpy as np
import pandas as pd
import pytest

from vivarium_public_health.metrics.utilities import (QueryString, OutputTemplate, to_years, get_output_template,
                                                      get_susceptible_person_time, get_disease_event_counts,
                                                      get_treatment_counts)


@pytest.fixture(params=((0, 100, 5, 1000), (20, 100, 5, 1000)))
def ages_and_bins(request):
    age_min = request.param[0]
    age_max = request.param[1]
    age_groups = request.param[2]
    num_ages = request.param[3]

    ages = np.linspace(age_min, age_max - age_groups/num_ages, num_ages)
    bin_ages, step = np.linspace(age_min, age_max, age_groups, endpoint=False, retstep=True)
    age_bins = pd.DataFrame({'age_group_start': bin_ages,
                             'age_group_end': bin_ages + step,
                             'age_group_name': [str(name) for name in range(len(bin_ages))]})

    return ages, age_bins


@pytest.fixture
def sexes():
    return ['Male', 'Female']


@pytest.fixture(params=list(product((True, False), repeat=3)))
def observer_config(request):
    c = {'by_age': request.param[0],
         'by_sex': request.param[1],
         'by_year': request.param[2]}
    return c


@pytest.mark.parametrize('reference, test', product([QueryString(''), QueryString('abc')], [QueryString(''), '']))
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


@pytest.mark.parametrize('a, b', product([QueryString('a')], [QueryString('b'), 'b']))
def test_query_string(a, b):
    assert a + b == 'a and b'
    assert a + b == QueryString('a and b')
    assert isinstance(a + b, QueryString)

    assert b + a == 'b and a'
    assert b + a == QueryString('b and a')
    assert isinstance(b + a, QueryString)

    a += b
    assert a == 'a and b'
    assert a == QueryString('a and b')
    assert isinstance(a, QueryString)

    b += a
    assert b == 'b and a and b'
    assert b == QueryString('b and a and b')
    assert isinstance(b, QueryString)


def test_get_output_template(observer_config):
    template = get_output_template(**observer_config)

    assert isinstance(template, OutputTemplate)
    assert '${measure}' in template.template

    if observer_config['by_year']:
        assert '_in_${year}' in template.template
    if observer_config['by_sex']:
        assert '_among_${sex}' in template.template
    if observer_config['by_age']:
        assert '_in_age_group_${age_group}' in template.template


@pytest.mark.parametrize('measure, sex, age, year',
                         product(['test', 'Test'], ['female', 'Female'],
                                 [1.0, 1, 'Early Neonatal'], [2011, '2011']))
def test_output_template(observer_config, measure, sex, age, year):
    template = get_output_template(**observer_config)

    out1 = template.substitute(measure=measure, sex=sex, age_group=age, year=year)
    out2 = template.substitute(measure=measure).substitute(sex=sex).substitute(age_group=age).substitute(year=year)
    assert out1 == out2


def test_output_template_exact():
    template = get_output_template(by_age=True, by_sex=True, by_year=True)

    out = template.substitute(measure='Test', sex='Female', age_group=1.0, year=2011)
    expected = 'test_in_2011_among_female_in_age_group_1.0'
    assert out == expected

    out = template.substitute(measure='Test', sex='Female', age_group='Early Neonatal', year=2011)
    expected = 'test_in_2011_among_female_in_age_group_early_neonatal'

    assert out == expected


def test_get_susceptible_person_time(ages_and_bins, sexes, observer_config):
    ages, age_bins = ages_and_bins
    disease = 'test_disease'
    states = [f'susceptible_to_{disease}', disease]
    pop = pd.DataFrame(list(product(ages, sexes, states)), columns=['age', 'sex', disease])
    pop['alive'] = 'alive'
    # Shuffle the rows
    pop = pop.sample(frac=1).reset_index(drop=True)

    year = 2017
    step_size = pd.Timedelta(days=7)

    person_time = get_susceptible_person_time(pop, observer_config, disease, year, step_size, age_bins)

    values = set(person_time.values())
    assert len(values) == 1
    expected_value = to_years(step_size)*len(pop)/2
    if observer_config['by_sex']:
        expected_value /= 2
    if observer_config['by_age']:
        expected_value /= len(age_bins)
    assert np.isclose(values.pop(), expected_value)

    # Doubling pop should double person time
    pop = pd.concat([pop, pop], axis=0, ignore_index=True)

    person_time = get_susceptible_person_time(pop, observer_config, disease, year, step_size, age_bins)

    values = set(person_time.values())
    assert len(values) == 1
    assert np.isclose(values.pop(), 2*expected_value)


def test_get_disease_event_counts(ages_and_bins, sexes, observer_config):
    ages, age_bins = ages_and_bins
    disease = 'test_disease'
    event_time = pd.Timestamp('1-1-2017')
    states = [event_time, pd.NaT]
    pop = pd.DataFrame(list(product(ages, sexes, states)), columns=['age', 'sex', f'{disease}_event_time'])
    # Shuffle the rows
    pop = pop.sample(frac=1).reset_index(drop=True)

    counts = get_disease_event_counts(pop, observer_config, disease, event_time, age_bins)

    values = set(counts.values())
    assert len(values) == 1
    expected_value = len(pop) / len(states)
    if observer_config['by_sex']:
        expected_value /= 2
    if observer_config['by_age']:
        expected_value /= len(age_bins)
    assert np.isclose(values.pop(), expected_value)

    # Doubling pop should double counts
    pop = pd.concat([pop, pop], axis=0, ignore_index=True)

    counts = get_disease_event_counts(pop, observer_config, disease, event_time, age_bins)

    values = set(counts.values())
    assert len(values) == 1
    assert np.isclose(values.pop(), 2 * expected_value)


def test_get_treatment_counts(ages_and_bins, sexes, observer_config):
    ages, age_bins = ages_and_bins
    treatment = 'test_treatment'
    event_time = pd.Timestamp('1-1-2017')
    dose_times = [event_time, event_time - pd.Timedelta(days=7)]
    doses = ['dose_1', 'dose_2']
    pop = pd.DataFrame(list(product(ages, sexes, dose_times, doses)),
                       columns=['age', 'sex', f'{treatment}_current_dose_event_time', f'{treatment}_current_dose'])
    # Shuffle the rows
    pop = pop.sample(frac=1).reset_index(drop=True)

    counts = get_treatment_counts(pop, observer_config, treatment, doses, event_time, age_bins)

    values = set(counts.values())
    assert len(values) == 1
    expected_value = len(pop) / (len(dose_times) * len(doses))
    if observer_config['by_sex']:
        expected_value /= 2
    if observer_config['by_age']:
        expected_value /= len(age_bins)
    assert np.isclose(values.pop(), expected_value)

    # Doubling pop should double counts
    pop = pd.concat([pop, pop], axis=0, ignore_index=True)

    counts = get_treatment_counts(pop, observer_config, treatment, doses, event_time, age_bins)

    values = set(counts.values())
    assert len(values) == 1
    assert np.isclose(values.pop(), 2 * expected_value)
