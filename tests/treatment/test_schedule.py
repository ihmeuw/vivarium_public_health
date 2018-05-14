import os

import pytest
import pandas as pd
import numpy as np
from scipy import stats

from ceam_public_health.treatment import TreatmentSchedule


@pytest.fixture(scope='function')
def config(base_config):
    metadata = {'layer': 'override', 'source': os.path.realpath(__file__)}
    base_config.update({
        'test_treatment': {
            'doses': ['first', 'second'],
            'dose_age_range': {
                'first': (36.5*2, 36.5*3),
                'second': (36.5*8, 36.5*8)
            },
        }
    }, **metadata)
    return base_config


@pytest.fixture(scope='function')
def builder(mocker, config):
    builder = mocker.MagicMock()
    builder.configuration = config
    return builder


@pytest.fixture(scope='function')
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
    tx = TreatmentSchedule('test_treatment')
    coverage = dict(first=1, second=0.5)
    tx.get_coverage = lambda builder_: coverage
    _current_schedule = mocker.Mock()
    with pytest.raises(NotImplementedError):
        tx._determine_dose_eligibility(builder, _current_schedule, pd.Index(range(5000)))
    with pytest.raises(NotImplementedError):
        tx.determine_dose_eligibility(builder, _current_schedule, pd.Index(range(5000)))
    tx.determine_dose_eligibility = mocker.Mock()
    eligible_index = pd.Index(range(0,5000,2))
    tx.determine_dose_eligibility.side_effect = lambda schedule_, dose_, index_: eligible_index

    assert np.array_equal(tx.determine_dose_eligibility(_current_schedule, 'second', pd.Index(range(5000))), eligible_index)


def test_determine_who_should_receive_dose_and_when(treatment_schedule, mocker):
    # pop size was chosen small since stats.normaltest (or other similar tests) tends to have smaller p-value for large
    # size population (https://stats.stackexchange.com/questions/2492/is-normality-testing-essentially-useless)
    # and I'm also sure that we can rely on CLT if we use large size sample.
    # if there's a better way to test normality, let me know. -MP

    pop = pd.DataFrame({'age': 500 * [0.1]})  # everybody at 36.5 days old
    treatment_schedule.population_view.get = mocker.Mock()
    treatment_schedule.population_view.get.return_value = pop
    doses = treatment_schedule.doses
    schedule = {dose: False for dose in doses}
    schedule.update({f'{dose}_age': np.NaN for dose in doses})
    schedule = pd.DataFrame(schedule, index=pop.index)
    dose_coverages = {'first': 1, 'second': 0.5}

    treatment_schedule.dose_coverages = lambda dose, index: pd.Series(dose_coverages[dose], index=index)
    treatment_schedule._determine_dose_eligibility = lambda _sch, _dose, index: pop.index

    coverage_draw = {'first': pd.Series(500 * [0.5]), 'second': pd.Series(250 * [0, 1])}


    def _who_should_receive_dose(simulant_data):
        population = treatment_schedule.population_view.get(simulant_data.index)
        dosed_pop = dict()
        for dose in doses:
            coverage = treatment_schedule.dose_coverages(dose, population.index)
            eligible_index = treatment_schedule._determine_dose_eligibility(schedule, dose, population.index)
            # this 'dosed_index' line is the main functionality to decide who should receive dose
            dosed_index = eligible_index[coverage_draw[dose][eligible_index] < coverage[eligible_index]]
            dosed_pop[dose] = dosed_index
        return dosed_pop

    def _when_should_receive_dose(simulant_data):
        dosed_pop = _who_should_receive_dose(simulant_data)
        population = treatment_schedule.population_view.get(simulant_data.index)
        for dose in doses:
            age_draw = pd.Series(np.random.rand(len(simulant_data)), index=population.index)
            min_age, max_age = treatment_schedule.dose_ages[dose]
            mean_age = (min_age + max_age) / 2
            age_std_dev = (mean_age - min_age) / 3
            age_at_dose = stats.norm(mean_age, age_std_dev).ppf(age_draw) \
                if age_std_dev else pd.Series(int(mean_age), index=population.index)

            age_at_dose[age_at_dose > max_age] = max_age * 0.99
            age_at_dose[age_at_dose < min_age] = min_age * 1.01

            schedule.loc[dosed_pop[dose], dose] = True
            schedule.loc[dosed_pop[dose], f'{dose}_age'] = age_at_dose[dosed_pop[dose]]
        return schedule

    # check whether it assigns the dose index properly
    dosed_pop = _who_should_receive_dose(pop)

    assert np.array_equal(dosed_pop['first'], coverage_draw['first'].index)
    assert np.array_equal(dosed_pop['second'], coverage_draw['second'][coverage_draw['second']==0].index)

    # check whether it scheduled them properly
    vaccine_schedule = _when_should_receive_dose(pop)
    z, pval = stats.normaltest(vaccine_schedule['first_age']) # first dose age should be approximately normal
    assert (pval > 0.05)  # p-value lower than 0.05 considered insignificant
    assert min(vaccine_schedule['first_age']) >= treatment_schedule.dose_ages['first'][0]
    assert max(vaccine_schedule['first_age']) <= treatment_schedule.dose_ages['first'][1]
    assert (vaccine_schedule['second_age'].loc[0] == treatment_schedule.dose_ages['second'][0] ==
            treatment_schedule.dose_ages['second'][1])
    # second dose should be scheduled for the same date as the first one that we checked above.
    assert (vaccine_schedule['second_age'].value_counts()[vaccine_schedule['second_age'].loc[0]] == 250)


def test_get_newly_dosed_simulants(treatment_schedule):
    tx = treatment_schedule
    pop = pd.DataFrame({'age': 5000 * [72.3/365]})
    pop[f'{tx.name}_current_dose'] = None
    schedule = {dose: False for dose in tx.doses}
    schedule.update({f'{dose}_age': np.NaN for dose in tx.doses})
    tx._schedule = pd.DataFrame(schedule, index=pop.index)
    tx._schedule['first'] = True
    tx._schedule['first_age'].loc[:2000] = 36.5*2  # 73days
    tx._schedule['first_age'].loc[2001:4000] = 36.5*2.5  # 91.25days
    tx._schedule['first_age'].loc[4001:]= 36.5*3  # 109.5 days
    tx._schedule['second'].loc[:2500] = True
    tx._schedule['second_age'].loc[:2500] = 36.5*8  # 292 days

    step_size = pd.Timedelta(days=1)

    assert np.array_equal(pop.loc[:2000], tx.get_newly_dosed_simulants('first', pop, step_size))

    pop['age'] = 90/365
    pop[f'{tx.name}_current_dose'].loc[:2000] = 'first'

    assert np.array_equal(pop.loc[2001:4000], tx.get_newly_dosed_simulants('first', pop, step_size))

    pop['age'] = 108.8/365
    pop[f'{tx.name}_current_dose'].loc[2001:4000] = 'first'

    assert np.array_equal(pop.loc[4001:], tx.get_newly_dosed_simulants('first', pop, step_size))

    pop['age'] = 291.3 / 365
    pop[f'{tx.name}_current_dose'].loc[4001:] = 'first'

    assert np.array_equal(pop.loc[:2500], tx.get_newly_dosed_simulants('second', pop, step_size))

