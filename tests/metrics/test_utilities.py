from itertools import product, combinations

import numpy as np
import pandas as pd
import pytest

from vivarium.testing_utilities import metadata
from vivarium_public_health.metrics.utilities import (QueryString, OutputTemplate, to_years, get_output_template,
                                                      get_susceptible_person_time, get_disease_event_counts,
                                                      get_age_sex_filter_and_iterables,
                                                      get_time_iterable, get_lived_in_span, get_person_time_in_span,
                                                      get_deaths, get_years_of_life_lost,
                                                      get_years_lived_with_disability, get_age_bins,
                                                      _MIN_YEAR, _MAX_YEAR, _MIN_AGE, _MAX_AGE)


@pytest.fixture(params=((0, 100, 5, 1000), (20, 100, 5, 1000)))
def ages_and_bins(request):
    age_min = request.param[0]
    age_max = request.param[1]
    age_groups = request.param[2]
    num_ages = request.param[3]

    ages = np.linspace(age_min, age_max - age_groups/num_ages, num_ages)
    bin_ages, step = np.linspace(age_min, age_max, age_groups, endpoint=False, retstep=True)
    age_bins = pd.DataFrame({'age_start': bin_ages,
                             'age_end': bin_ages + step,
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


@pytest.fixture()
def builder(mocker):
    builder = mocker.MagicMock()
    df = pd.DataFrame({'age_start': [0, 1, 4],
                       'age_group_name': ['youngest', 'younger', 'young'],
                       'age_end': [1, 4, 6]})
    builder.data.load.return_value = df
    return builder


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
        assert age_sex_filter == '{age_start} <= age and age < {age_end} and sex == "{sex}"'

        for (g1, s1), (g2, s2) in zip(ages, age_bins.set_index('age_group_name').iterrows()):
            assert g1 == g2
            assert s1.equals(s2)

        assert sexes == ['Male', 'Female']

    elif observer_config['by_age']:
        assert age_sex_filter == '{age_start} <= age and age < {age_end}'

        for (g1, s1), (g2, s2) in zip(ages, age_bins.set_index('age_group_name').iterrows()):
            assert g1 == g2
            assert s1.equals(s2)

        assert sexes == ['Both']
    elif observer_config['by_sex']:
        assert age_sex_filter == 'sex == "{sex}"'

        assert len(ages) == 1
        group, data = ages[0]
        assert group == 'all_ages'
        assert data['age_start'] == _MIN_AGE
        assert data['age_end'] == _MAX_AGE

        assert sexes == ['Male', 'Female']

    else:
        assert age_sex_filter == ''

        assert len(ages) == 1
        group, data = ages[0]
        assert group == 'all_ages'
        assert data['age_start'] == _MIN_AGE
        assert data['age_end'] == _MAX_AGE

        assert sexes == ['Both']


def test_get_age_sex_filter_and_iterables_with_span(ages_and_bins, observer_config):
    _, age_bins = ages_and_bins
    age_sex_filter, (ages, sexes) = get_age_sex_filter_and_iterables(observer_config, age_bins, in_span=True)

    assert isinstance(age_sex_filter, QueryString)
    if observer_config['by_age'] and observer_config['by_sex']:
        expected = '{age_start} < age_at_span_end and age_at_span_start < {age_end} and sex == "{sex}"'
        assert age_sex_filter == expected

        for (g1, s1), (g2, s2) in zip(ages, age_bins.set_index('age_group_name').iterrows()):
            assert g1 == g2
            assert s1.equals(s2)

        assert sexes == ['Male', 'Female']

    elif observer_config['by_age']:
        assert age_sex_filter == '{age_start} < age_at_span_end and age_at_span_start < {age_end}'

        for (g1, s1), (g2, s2) in zip(ages, age_bins.set_index('age_group_name').iterrows()):
            assert g1 == g2
            assert s1.equals(s2)

        assert sexes == ['Both']
    elif observer_config['by_sex']:
        assert age_sex_filter == 'sex == "{sex}"'

        assert len(ages) == 1
        group, data = ages[0]
        assert group == 'all_ages'
        assert data['age_start'] == _MIN_AGE
        assert data['age_end'] == _MAX_AGE

        assert sexes == ['Male', 'Female']

    else:
        assert age_sex_filter == ''

        assert len(ages) == 1
        group, data = ages[0]
        assert group == 'all_ages'
        assert data['age_start'] == _MIN_AGE
        assert data['age_end'] == _MAX_AGE

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
    assert start == pd.Timestamp(f'1-1-{_MIN_YEAR}')
    assert end == pd.Timestamp(f'1-1-{_MAX_YEAR}')


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
        assert end == pd.Timestamp(f'1-1-{year+1}')


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


def test_get_person_time_in_span(ages_and_bins, observer_config):
    _, age_bins = ages_and_bins
    start = int(age_bins.age_start.min())
    end = int(age_bins.age_end.max())
    n_ages = len(list(range(start, end)))
    n_bins = len(age_bins)
    segments_per_age = [(i + 1)*(n_ages - i) for i in range(n_ages)]
    ages_per_bin = n_ages // n_bins
    age_bins['expected_time'] = [sum(segments_per_age[ages_per_bin*i:ages_per_bin*(i+1)]) for i in range(n_bins)]

    age_starts, age_ends = zip(*combinations(range(start, end + 1), 2))
    women = pd.DataFrame({'age_at_span_start': age_starts, 'age_at_span_end': age_ends, 'sex': 'Female'})
    men = women.copy()
    men.loc[:, 'sex'] = 'Male'

    lived_in_span = pd.concat([women, men], ignore_index=True).sample(frac=1).reset_index(drop=True)
    base_filter = QueryString("")
    span_key = get_output_template(**observer_config).substitute(measure='person_time', year=2019)

    pt = get_person_time_in_span(lived_in_span, base_filter, span_key, observer_config, age_bins)

    if observer_config['by_age']:
        for group, age_bin in age_bins.iterrows():
            group_pt = sum(set([v for k, v in pt.items() if f'in_age_group_{group}' in k]))
            if observer_config['by_sex']:
                assert group_pt == age_bin.expected_time
            else:
                assert group_pt == 2 * age_bin.expected_time
    else:
        group_pt = sum(set(pt.values()))
        if observer_config['by_sex']:
            assert group_pt == age_bins.expected_time.sum()
        else:
            assert group_pt == 2 * age_bins.expected_time.sum()


def test_get_deaths(ages_and_bins, sexes, observer_config):
    alive = ['dead', 'alive']
    ages, age_bins = ages_and_bins
    exit_times = [pd.Timestamp('1-1-2012'), pd.Timestamp('1-1-2013')]
    causes = ['cause_a', 'cause_b']

    pop = pd.DataFrame(list(product(alive, ages, sexes, exit_times, causes)),
                       columns=['alive', 'age', 'sex', 'exit_time', 'cause_of_death'])
    # Shuffle the rows
    pop = pop.sample(frac=1).reset_index(drop=True)

    deaths = get_deaths(pop, observer_config, pd.Timestamp('1-1-2010'), pd.Timestamp('1-1-2015'), age_bins, causes)
    values = set(deaths.values())

    expected_value = len(pop) / (len(causes) * len(alive))
    if observer_config['by_year']:
        assert len(values) == 2  # Uniform across bins with deaths, 0 in year bins without deaths
        expected_value /= 2
    else:
        assert len(values) == 1
    value = max(values)
    if observer_config['by_sex']:
        expected_value /= 2
    if observer_config['by_age']:
        expected_value /= len(age_bins)
    assert np.isclose(value, expected_value)

    # Doubling pop should double counts
    pop = pd.concat([pop, pop], axis=0, ignore_index=True)

    deaths = get_deaths(pop, observer_config, pd.Timestamp('1-1-2010'), pd.Timestamp('1-1-2015'), age_bins, causes)
    values = set(deaths.values())

    if observer_config['by_year']:
        assert len(values) == 2  # Uniform across bins with deaths, 0 in year bins without deaths
    else:
        assert len(values) == 1
    value = max(values)
    assert np.isclose(value, 2 * expected_value)


def test_get_years_of_life_lost(ages_and_bins, sexes, observer_config):
    alive = ['dead', 'alive']
    ages, age_bins = ages_and_bins
    exit_times = [pd.Timestamp('1-1-2012'), pd.Timestamp('1-1-2013')]
    causes = ['cause_a', 'cause_b']

    pop = pd.DataFrame(list(product(alive, ages, sexes, exit_times, causes)),
                       columns=['alive', 'age', 'sex', 'exit_time', 'cause_of_death'])
    # Shuffle the rows
    pop = pop.sample(frac=1).reset_index(drop=True)

    def life_expectancy(index):
        return pd.Series(1, index=index)

    ylls = get_years_of_life_lost(pop, observer_config, pd.Timestamp('1-1-2010'), pd.Timestamp('1-1-2015'),
                                  age_bins, life_expectancy, causes)
    values = set(ylls.values())

    expected_value = len(pop) / (len(causes) * len(alive))
    if observer_config['by_year']:
        assert len(values) == 2  # Uniform across bins with deaths, 0 in year bins without deaths
        expected_value /= 2
    else:
        assert len(values) == 1
    value = max(values)
    if observer_config['by_sex']:
        expected_value /= 2
    if observer_config['by_age']:
        expected_value /= len(age_bins)
    assert np.isclose(value, expected_value)

    # Doubling pop should double counts
    pop = pd.concat([pop, pop], axis=0, ignore_index=True)

    ylls = get_years_of_life_lost(pop, observer_config, pd.Timestamp('1-1-2010'), pd.Timestamp('1-1-2015'),
                                  age_bins, life_expectancy, causes)
    values = set(ylls.values())

    if observer_config['by_year']:
        assert len(values) == 2  # Uniform across bins with deaths, 0 in year bins without deaths
    else:
        assert len(values) == 1
    value = max(values)
    assert np.isclose(value, 2 * expected_value)


def test_get_years_lived_with_disability(ages_and_bins, sexes, observer_config):
    alive = ['dead', 'alive']
    ages, age_bins = ages_and_bins
    causes = ['cause_a', 'cause_b']
    cause_a = ['susceptible_to_cause_a', 'cause_a']
    cause_b = ['susceptible_to_cause_b', 'cause_b']
    year = 2010
    step_size = pd.Timedelta(days=7)

    pop = pd.DataFrame(list(product(alive, ages, sexes, cause_a, cause_b)),
                       columns=['alive', 'age', 'sex'] + causes)
    # Shuffle the rows
    pop = pop.sample(frac=1).reset_index(drop=True)

    def disability_weight(cause):
        def inner(index):
            sub_pop = pop.loc[index]
            return pd.Series(1, index=index) * (sub_pop[cause] == cause)
        return inner

    disability_weights = {cause: disability_weight(cause) for cause in causes}

    ylds = get_years_lived_with_disability(pop, observer_config, year, step_size, age_bins, disability_weights, causes)

    values = set(ylds.values())
    assert len(values) == 1
    states_per_cause = len(cause_a)
    expected_value = len(pop) / (len(alive) * states_per_cause) * to_years(step_size)
    if observer_config['by_sex']:
        expected_value /= 2
    if observer_config['by_age']:
        expected_value /= len(age_bins)
    assert np.isclose(values.pop(), expected_value)

    # Doubling pop should double person time
    pop = pd.concat([pop, pop], axis=0, ignore_index=True)

    ylds = get_years_lived_with_disability(pop, observer_config, year, step_size, age_bins, disability_weights, causes)

    values = set(ylds.values())
    assert len(values) == 1
    assert np.isclose(values.pop(), 2 * expected_value)


@pytest.mark.parametrize('age_start, exit_age, result_age_end_values, result_age_start_values',
                         [(2, 5, {4, 5}, {2, 4}),
                          (0, None, {1, 4, 6}, {0, 1, 4}),
                          (1, 4, {4}, {1}),
                          (1, 3, {3}, {1}),
                          (0.8, 6, {1, 4, 6}, {0.8, 1, 4})])
def test_get_age_bins(builder, base_config, age_start, exit_age, result_age_end_values, result_age_start_values):
    base_config.update({
        'population': {
            'age_start': age_start,
            'exit_age': exit_age
        }
    }, **metadata(__file__))
    builder.configuration = base_config
    df = get_age_bins(builder)
    assert set(df.age_end) == result_age_end_values
    assert set(df.age_start) == result_age_start_values

