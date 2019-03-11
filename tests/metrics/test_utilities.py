from itertools import product, combinations

import numpy as np
import pandas as pd
import pytest

from vivarium_public_health.metrics.utilities import (QueryString, OutputTemplate, to_years, get_output_template,
                                                      get_susceptible_person_time, get_disease_event_counts,
                                                      get_treatment_counts, get_age_sex_filter_and_iterables,
                                                      get_time_iterable, get_lived_in_span)


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


def test_get_age_sex_filter_and_iterables(ages_and_bins, observer_config):
    _, age_bins = ages_and_bins
    age_sex_filter, (ages, sexes) = get_age_sex_filter_and_iterables(observer_config, age_bins)

    assert isinstance(age_sex_filter, QueryString)
    if observer_config['by_age'] and observer_config['by_sex']:
        assert age_sex_filter == '{age_group_start} <= age and age < {age_group_end} and sex == "{sex}"'

        for (g1, s1), (g2, s2) in zip(ages, age_bins.set_index('age_group_name').iterrows()):
            assert g1 == g2
            assert s1.equals(s2)

        assert sexes == ['Male', 'Female']

    elif observer_config['by_age']:
        assert age_sex_filter == '{age_group_start} <= age and age < {age_group_end}'

        for (g1, s1), (g2, s2) in zip(ages, age_bins.set_index('age_group_name').iterrows()):
            assert g1 == g2
            assert s1.equals(s2)

        assert sexes == ['Both']
    elif observer_config['by_sex']:
        assert age_sex_filter == 'sex == "{sex}"'

        assert len(ages) == 1
        group, data = ages[0]
        assert group == 'all_ages'
        assert data['age_group_start'] is None
        assert data['age_group_end'] is None

        assert sexes == ['Male', 'Female']

    else:
        assert age_sex_filter == ''

        assert len(ages) == 1
        group, data = ages[0]
        assert group == 'all_ages'
        assert data['age_group_start'] is None
        assert data['age_group_end'] is None

        assert sexes == ['Both']


def test_get_age_sex_filter_and_iterables_with_span(ages_and_bins, observer_config):
    _, age_bins = ages_and_bins
    age_sex_filter, (ages, sexes) = get_age_sex_filter_and_iterables(observer_config, age_bins, in_span=True)

    assert isinstance(age_sex_filter, QueryString)
    if observer_config['by_age'] and observer_config['by_sex']:
        expected = '{age_group_start} < age_at_span_end and age_at_span_start < {age_group_end} and sex == "{sex}"'
        assert age_sex_filter == expected

        for (g1, s1), (g2, s2) in zip(ages, age_bins.set_index('age_group_name').iterrows()):
            assert g1 == g2
            assert s1.equals(s2)

        assert sexes == ['Male', 'Female']

    elif observer_config['by_age']:
        assert age_sex_filter == '{age_group_start} < age_at_span_end and age_at_span_start < {age_group_end}'

        for (g1, s1), (g2, s2) in zip(ages, age_bins.set_index('age_group_name').iterrows()):
            assert g1 == g2
            assert s1.equals(s2)

        assert sexes == ['Both']
    elif observer_config['by_sex']:
        assert age_sex_filter == 'sex == "{sex}"'

        assert len(ages) == 1
        group, data = ages[0]
        assert group == 'all_ages'
        assert data['age_group_start'] is None
        assert data['age_group_end'] is None

        assert sexes == ['Male', 'Female']

    else:
        assert age_sex_filter == ''

        assert len(ages) == 1
        group, data = ages[0]
        assert group == 'all_ages'
        assert data['age_group_start'] is None
        assert data['age_group_end'] is None

        assert sexes == ['Both']


@pytest.mark.parametrize('year_start, year_end', [(2011, 2017), (2011, 2011)])
def test_get_time_iterable_no_year(year_start, year_end):
    config = {'by_year': False}
    sim_start = pd.Timestamp(f'7-2-{year_start}')
    sim_end = pd.Timestamp(f'3-15-{year_end}')

    time_spans = get_time_iterable(config, sim_start, sim_end)

    assert len(time_spans) == 1
    name, (start, end) = time_spans[0]
    assert name == 'all_years'
    assert start == pd.Timestamp('1-1-1900')
    assert end == pd.Timestamp('1-1-2100')


@pytest.mark.parametrize('year_start, year_end', [(2011, 2017), (2011, 2011)])
def test_get_time_iterable_with_year(year_start, year_end):
    config = {'by_year': True}
    sim_start = pd.Timestamp(f'7-2-{year_start}')
    sim_end = pd.Timestamp(f'3-15-{year_end}')

    time_spans = get_time_iterable(config, sim_start, sim_end)

    years = list(range(year_start, year_end + 1))
    assert len(time_spans) == len(years)
    for year, time_span in zip(years, time_spans):
        name, (start, end) = time_span
        assert name == year
        assert start == pd.Timestamp(f'1-1-{year}')
        assert end  == pd.Timestamp(f'1-1-{year+1}')


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


def test_get_lived_in_span():
    dt = pd.Timedelta(days=5)
    reference_t = pd.Timestamp('1-10-2010')

    early_1 = reference_t - 2*dt
    early_2 = reference_t - dt

    t_start = reference_t

    mid_1 = reference_t + dt
    mid_2 = reference_t + 2*dt

    t_end = reference_t + 3*dt

    late_1 = reference_t + 4*dt
    late_2 = reference_t + 5*dt

    # 28 combinations, six of which are entirely out of the time span
    times = [early_1, early_2, t_start, mid_1, mid_2, t_end, late_1, late_2]
    starts, ends = zip(*combinations(times, 2))
    pop = pd.DataFrame({'age': to_years(10*dt), 'entrance_time': starts, 'exit_time': ends})

    lived_in_span = get_lived_in_span(pop, t_start, t_end)
    # Indices here are from the combinatorics math. They represent
    # 0: (early_1, early_2)
    # 1: (early_1, t_start)
    # 7: (early_2, t_start)
    # 25: (t_end, late_1)
    # 26: (t_end, late_2)
    # 27: (late_1, late_2)
    assert {0, 1, 7, 25, 26, 27}.intersection(lived_in_span.index) == set()

    exit_before_span_end = lived_in_span.exit_time <= t_end
    assert np.all(lived_in_span.loc[exit_before_span_end, 'age_at_span_end']
                  == lived_in_span.loc[exit_before_span_end, 'age'])

    exit_after_span_end = ~exit_before_span_end
    age_at_end = lived_in_span.age - to_years(lived_in_span.exit_time - t_end)
    assert np.all(lived_in_span.loc[exit_after_span_end, 'age_at_span_end']
                  == age_at_end.loc[exit_after_span_end])

    enter_after_span_start = lived_in_span.entrance_time >= t_start
    age_at_start = lived_in_span.age - to_years(lived_in_span.exit_time - lived_in_span.entrance_time)
    assert np.all(lived_in_span.loc[enter_after_span_start, 'age_at_span_start']
                  == age_at_start.loc[enter_after_span_start])

    enter_before_span_start = ~enter_after_span_start
    age_at_start = lived_in_span.age - to_years(lived_in_span.exit_time - t_start)
    assert np.all(lived_in_span.loc[enter_before_span_start, 'age_at_span_start']
                  == age_at_start.loc[enter_before_span_start])


def test_get_lived_in_span_no_one_in_span():
    dt = pd.Timedelta(days=365.25)
    t_start = pd.Timestamp('1-1-2010')
    t_end = t_start + dt

    pop = pd.DataFrame({'entrance_time': t_start - 2*dt, 'exit_time': t_start - dt, 'age': range(100)})
    lived_in_span = get_lived_in_span(pop, t_start, t_end)
    assert lived_in_span.empty

    pop = pd.DataFrame({'entrance_time': t_end + dt, 'exit_time': t_end + 2*dt, 'age': range(100)})
    lived_in_span = get_lived_in_span(pop, t_start, t_end)
    assert lived_in_span.empty
