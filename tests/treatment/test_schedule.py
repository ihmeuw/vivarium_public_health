import os

import pytest
import pandas as pd
import numpy as np

from vivarium_public_health.treatment import TreatmentSchedule


@pytest.fixture
def config(base_config):
    metadata = {'layer': 'override', 'source': os.path.realpath(__file__)}
    base_config.update({
        'test_treatment': {
            'doses': ['first', 'second'],
            'dose_age_range': {
                'first': {
                    'start': 60,
                    'end': 90,
                },
                'second': {
                    'start': 180,
                    'end': 180,
                },
            },
        }
    }, **metadata)
    return base_config


@pytest.fixture
def builder(mocker, config):
    builder = mocker.MagicMock()
    builder.configuration = config
    return builder


@pytest.fixture
def treatment_schedule(mocker, builder):
    tx = TreatmentSchedule('test_treatment')
    coverage = dict(first=1, second=0.5)
    tx.get_coverage = lambda builder_: coverage
    tx.determine_dose_eligibility = mocker.Mock()
    tx.setup(builder)
    return tx


def test_get_coverage(builder, mocker):
    tx = TreatmentSchedule('test_treatment')
    tx.determine_dose_eligibility = mocker.MagicMock()

    with pytest.raises(NotImplementedError):
        tx._get_coverage(builder)
    with pytest.raises(NotImplementedError):
        tx.get_coverage(builder)

    coverage = dict(first=1, second=0.5)
    tx.get_coverage = lambda builder_: coverage

    assert tx.get_coverage(builder) == coverage


def test_determine_dose_eligibility(builder, mocker):
    population_size = 5000
    tx = TreatmentSchedule('test_treatment')
    coverage = dict(first=1, second=0.5)
    tx.get_coverage = lambda builder_: coverage
    _current_schedule = mocker.Mock()

    with pytest.raises(NotImplementedError):
        tx._determine_dose_eligibility(builder, _current_schedule, pd.Index(range(population_size)))
    with pytest.raises(NotImplementedError):
        tx.determine_dose_eligibility(builder, _current_schedule, pd.Index(range(population_size)))

    tx.determine_dose_eligibility = mocker.Mock()
    eligible_index = pd.Index(range(0, population_size, 2))
    tx.determine_dose_eligibility.side_effect = lambda schedule_, dose_, index_: eligible_index

    assert np.array_equal(tx.determine_dose_eligibility(_current_schedule, 'second', pd.Index(range(population_size))),
                          eligible_index)


def test_who_should_receive_dose(treatment_schedule, mocker):
    population_size = 5000
    age = 0.1
    pop = pd.DataFrame({'age': population_size*[age]})
    tx = treatment_schedule
    tx.population_view.get = mocker.Mock()
    tx.population_view.get.return_value = pop
    doses = ['first', 'second']

    coverage = {'first': lambda index: pd.Series(1, index=index), 'second': lambda index: pd.Series(0.5, index=index)}
    tx.dose_coverages = coverage

    tx._determine_dose_eligibility = lambda _sch, _dose, index: pop.index

    schedule = {dose: False for dose in doses}
    schedule.update({f'{dose}_age': np.NaN for dose in doses})

    tx._determine_dose_eligibility = lambda _sch, _dose, index: pop.index

    def coverage_draw_side_effect(index, additional_key):
        if additional_key == 'first_covered':
            return pd.Series(0.5, index=index)
        else:
            coverage_draws = pd.Series(0, index=index)
            coverage_draws.loc[len(index)//2:] = 1
            return coverage_draws

    tx.randomness.get_draw.side_effect = coverage_draw_side_effect

    first_dosed_index = tx._determine_who_should_receive_dose('first', pop, schedule)
    assert np.array_equal(first_dosed_index, pop.index)

    second_dosed_index = tx._determine_who_should_receive_dose('second', pop, schedule)
    assert np.array_equal(second_dosed_index, pd.Index(range(len(pop.index)//2)))


def test_when_should_receive_dose(treatment_schedule, mocker):
    population_size = 5000
    age = 0.1
    pop = pd.DataFrame({'age': population_size*[age]})
    tx = treatment_schedule
    tx.population_view.get = mocker.Mock()
    tx.population_view.get.return_value = pop

    def age_draw_side_effect(index, additional_key):
        age_draws = [0.00001, 0.25, 0.5, 0.75, 0.99999]  # should give ages < min_age and age > max_age
        if len(index) < len(age_draws):
            return pd.Series(age_draws[:index])

        else:
            return pd.Series(age_draws * (len(index)//5) + age_draws[:(len(index) % 5)])

    tx.randomness.get_draw.side_effect = age_draw_side_effect

    # any ages outsider [min_age, max_age] should be pushed inside of the ranage.
    age_at_first_dose = tx._determine_when_should_receive_dose('first', pop)
    assert min(age_at_first_dose) == tx.dose_ages['first']['start'] * 1.01
    assert max(age_at_first_dose) == tx.dose_ages['first']['end'] * 0.99

    # min_age and max_age are same, should be all equal to a single age
    age_at_second_dose = tx._determine_when_should_receive_dose('second', pop)
    assert min(age_at_second_dose) == tx.dose_ages['second']['start']
    assert max(age_at_second_dose) == tx.dose_ages['second']['end']


def test_get_newly_dosed_simulants(treatment_schedule):
    population_size = 5000
    tx = treatment_schedule
    age = 59.3
    pop = pd.DataFrame({'age': population_size * [age/365]})
    pop[f'{tx.name}_current_dose'] = None
    schedule = {dose: False for dose in tx.doses}
    schedule.update({f'{dose}_age': np.NaN for dose in tx.doses})
    tx._schedule = pd.DataFrame(schedule, index=pop.index)
    tx._schedule['first'] = True

    # first dose at 60 days
    tx._schedule.loc[:1999, 'first_age'] = 60

    # first dose at 75 days
    tx._schedule.loc[2000:3999, 'first_age'] = 75

    # first dose at 90 days
    tx._schedule.loc[4000:, 'first_age'] = 90

    # second dose True
    tx._schedule.loc[:2500, 'second'] = True
    tx._schedule.loc[:2500, 'second_age'] = 180

    step_size = pd.Timedelta(days=1)

    # current age = 59.3 days
    assert np.array_equal(pop[tx._schedule.first_age == 60], tx.get_newly_dosed_simulants('first', pop, step_size))
    pop.loc[tx._schedule.first_age == 60, f'{tx.name}_current_dose'] = 'first'  # update the current dose

    # current age = 74.3 days
    age = 74.3  # days
    pop['age'] = age/365

    assert np.array_equal(pop[tx._schedule.first_age == 75], tx.get_newly_dosed_simulants('first', pop, step_size))
    pop.loc[tx._schedule.first_age == 75, f'{tx.name}_current_dose'] = 'first'  # update the current dose

    # current age = 89.3 days
    age = 89.3  # days
    pop['age'] = age/365

    assert np.array_equal(pop[tx._schedule.first_age == 90], tx.get_newly_dosed_simulants('first', pop, step_size))
    pop.loc[tx._schedule.first_age == 90, f'{tx.name}_current_dose'] = 'first'  # update the current dose

    # current age =179.3 days, ready for second dose
    age = 179.3  # days
    pop['age'] = age / 365

    assert np.array_equal(pop[tx._schedule.second == True], tx.get_newly_dosed_simulants('second', pop, step_size))
    pop.loc[tx._schedule.second == True, f'{tx.name}_current_dose'] = 'second'  # update the current dose
