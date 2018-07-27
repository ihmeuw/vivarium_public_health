import pytest
import pandas as pd
import numpy as np

from vivarium.framework.components import ComponentConfigError
from vivarium.testing_utilities import metadata

from vivarium_public_health.treatment import Treatment


@pytest.fixture
def config(base_config):
    base_config.update({
        'test_treatment': {
            'dose_response': {
                'onset_delay': 14,  # Days
                'duration': 360,  # Days
                'waning_rate': 0.038  # Percent/Day
            },
        }
    }, **metadata(__file__))
    return base_config


@pytest.fixture
def test_population():
    tx = Treatment('test_treatment', 'test_cause')
    cols = ['active_dose',
            'immunity',
            f'{tx.name}_current_dose',
            f'{tx.name}_current_dose_event_time',
            f'{tx.name}_previous_dose',
            f'{tx.name}_previous_dose_event_time']

    # grp 1 does not get any dose and get no immunity
    grp1 = pd.DataFrame({'active_dose': None, 'immunity': None}, columns=cols, index=range(1000))

    # grp 2 got only one current dose and get full immunity from it
    grp2 = pd.DataFrame({'active_dose': 'first',
                         'immunity': 'full',
                         f'{tx.name}_current_dose': 'first',
                         f'{tx.name}_current_dose_event_time': pd.Timestamp('06-15-2005')}, index=range(1000, 2000))
    # grp 3 got only one current dose and get waning immunity from it
    grp3 = pd.DataFrame({'active_dose': 'first',
                         'immunity': 'waning',
                         f'{tx.name}_current_dose': 'first',
                         f'{tx.name}_current_dose_event_time': pd.Timestamp('06-15-2004')}, index=range(2000, 3000))
    # grp 4 got only one current dose but still do not get immunity from it
    grp4 = pd.DataFrame({'active_dose': None,
                         'immunity': None,
                         f'{tx.name}_current_dose': 'first',
                         f'{tx.name}_current_dose_event_time': pd.Timestamp('06-30-2005')}, index=range(3000, 4000))

    # grp 5 got both current and previous doses and get full immunity from current dose
    grp5 = pd.DataFrame({'active_dose': 'second',
                         'immunity': 'full',
                         f'{tx.name}_current_dose': 'second',
                         f'{tx.name}_current_dose_event_time': pd.Timestamp('06-15-2005'),
                         f'{tx.name}_previous_dose': 'first',
                         f'{tx.name}_previous_dose_event_time': pd.Timestamp('12-15-2004')}, index=range(4000, 5000))

    # grp 6 got both current and previous doses and get waning immunity from current dose
    grp6 = pd.DataFrame({'active_dose': 'second',
                         'immunity': 'waning',
                         f'{tx.name}_current_dose': 'second',
                         f'{tx.name}_current_dose_event_time': pd.Timestamp('06-20-2004'),
                         f'{tx.name}_previous_dose': 'first',
                         f'{tx.name}_previous_dose_event_time': pd.Timestamp('12-15-2003')}, index=range(5000, 6000))

    # grp 7 got both current and previous doses but get full immunity from previous dose
    grp7 = pd.DataFrame({'active_dose': 'first',
                         'immunity': 'full',
                         f'{tx.name}_current_dose': 'second',
                         f'{tx.name}_current_dose_event_time': pd.Timestamp('06-30-2005'),
                         f'{tx.name}_previous_dose': 'first',
                         f'{tx.name}_previous_dose_event_time': pd.Timestamp('01-01-2005')}, index=range(6000, 7000))

    pop = pd.concat([grp1, grp2, grp3, grp4, grp5, grp6, grp7])
    return pop


@pytest.fixture
def builder(mocker, config):
    builder = mocker.MagicMock()
    builder.configuration = config
    return builder


@pytest.fixture
def treatment(builder):
    tx = Treatment('test_treatment', 'test_cause')

    protection = {'first': 0.5, 'second': 0.7}
    tx.get_protection = lambda builder_: protection

    tx.setup(builder)

    tx.clock = lambda: pd.Timestamp('07-02-2005')
    return tx


def test_setup(builder):
    tx = Treatment('not_a_treatment', 'test_cause')

    with pytest.raises(ComponentConfigError):
        tx.setup(builder)

    tx = Treatment('test_treatment', 'test_cause')

    with pytest.raises(NotImplementedError):
        tx.setup(builder)


def test_get_protection(builder):
    tx = Treatment('test_treatment', 'test_cause')

    with pytest.raises(NotImplementedError):
        tx._get_protection(builder)

    with pytest.raises(NotImplementedError):
        tx.get_protection(builder)

    protection = {'first': 0.5, 'second': 0.7}
    tx.get_protection = lambda builder_: protection

    assert tx._get_protection(builder) == protection


def test_get_dosing_status(treatment, test_population):
    tx = treatment
    dosing_status = tx._get_dosing_status(test_population)

    expected_dosing_status = pd.DataFrame({'dose': None, 'date': pd.NaT}, index=test_population.index)

    expected_dosing_status['dose'] = test_population['active_dose']
    current_dose_active = test_population['active_dose'] == test_population[f'{tx.name}_current_dose']
    previous_dose_active = test_population['active_dose'] == test_population[f'{tx.name}_previous_dose']
    expected_dosing_status.loc[current_dose_active, 'date'] = test_population[f'{tx.name}_current_dose_event_time']
    expected_dosing_status.loc[previous_dose_active, 'date'] = test_population[f'{tx.name}_previous_dose_event_time']

    no_immunity = (test_population['active_dose'].isna()) & (test_population['immunity'].isna())
    assert pd.DataFrame.equals(expected_dosing_status[no_immunity], dosing_status[no_immunity])

    first_full_immunity = (test_population['active_dose'] == 'first') & (test_population['immunity'] == 'full')
    assert pd.DataFrame.equals(expected_dosing_status[first_full_immunity], dosing_status[first_full_immunity])

    first_waning_immunity = (test_population['active_dose'] == 'first') & (test_population['immunity'] == 'waning')
    assert pd.DataFrame.equals(expected_dosing_status[first_waning_immunity], dosing_status[first_waning_immunity])

    second_full_immunity = (test_population['active_dose'] == 'second') & (test_population['immunity'] == 'full')
    assert pd.DataFrame.equals(expected_dosing_status[second_full_immunity], dosing_status[second_full_immunity])

    second_waning_immunity = (test_population['active_dose'] == 'second') & (test_population['immunity'] == 'waning')
    assert pd.DataFrame.equals(expected_dosing_status[second_waning_immunity], dosing_status[second_waning_immunity])


def test_determine_protection(treatment, test_population):
    tx = treatment

    # No immunity : grp1, grp4
    expected_protection = pd.Series(0, index=test_population.index)

    # First dose full immunity
    first_full_immunity = (test_population['active_dose'] == 'first') & (test_population['immunity'] == 'full')
    expected_protection[first_full_immunity] = tx.protection['first']

    # Second dose full immunity]
    second_full_immunity = (test_population['active_dose'] == 'second') & (test_population['immunity'] == 'full')
    expected_protection[second_full_immunity] = tx.protection['second']

    # First dose waning immunity (time in waning 8 days)
    first_waning_immunity = (test_population['active_dose'] == 'first') & (test_population['immunity'] == 'waning')
    expected_protection[first_waning_immunity] = (tx.protection['first']
                                                  * np.exp(-tx.dose_response['waning_rate'] * 8))

    # Second dose waning immunity (time in waning 3 days)
    second_waning_immunity = (test_population['active_dose'] == 'second') & (test_population['immunity'] == 'waning')
    expected_protection[second_waning_immunity] = (tx.protection['second']
                                                   * np.exp(-tx.dose_response['waning_rate'] * 3))

    protection = tx.determine_protection(test_population)

    assert pd.DataFrame.equals(expected_protection[first_full_immunity], protection[first_full_immunity])
    assert pd.DataFrame.equals(expected_protection[first_waning_immunity], protection[first_waning_immunity])
    assert pd.DataFrame.equals(expected_protection[second_full_immunity], protection[second_full_immunity])
    assert pd.DataFrame.equals(expected_protection[second_waning_immunity], protection[second_waning_immunity])


def test_incidence_rates(treatment, mocker):
    protection = 0.5
    base_rate = 1

    treatment.determine_protection = mocker.Mock()
    treatment.determine_protection.side_effect = lambda index: pd.Series(protection, index=index)

    treatment.population_view.get = mocker.Mock()
    population_size = 5000
    return_pop = pd.DataFrame({'alive': population_size*['alive', 'dead', 'untracked']})
    alive_pop = return_pop[return_pop.alive == 'alive']
    treatment.population_view.get.return_value = return_pop
    rates = pd.Series(base_rate, index=alive_pop.index)

    incidence = treatment.incidence_rates(return_pop.index, rates)
    assert np.all(incidence == base_rate*(1-protection))
