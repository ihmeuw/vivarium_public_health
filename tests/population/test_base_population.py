import os
import math
from itertools import product

import numpy as np
import pandas as pd
import pytest

from vivarium.test_util import setup_simulation, pump_simulation, build_table, get_randomness

import ceam_public_health.population.base_population as bp
import ceam_public_health.population.data_transformations as dt


@pytest.fixture(scope='function')
def config(base_config):
    try:
        base_config.reset_layer('override', preserve_keys=['input_data.intermediary_data_cache_path',
                                                           'input_data.auxiliary_data_folder'])
    except KeyError:
        pass

    metadata = {'layer': 'override', 'source': os.path.realpath(__file__)}
    base_config.population.set_with_metadata('age_start', 0, **metadata)
    base_config.population.set_with_metadata('age_end', 110, **metadata)
    base_config.population.set_with_metadata('use_subregions', False, **metadata)
    base_config.input_data.set_with_metadata('location', 180, **metadata)
    return base_config


@pytest.fixture(scope='function')
def build_pop_data_table_mock(mocker):
    return mocker.patch('ceam_public_health.population.base_population._build_population_data_table')


@pytest.fixture(scope='function')
def generate_ceam_population_mock(mocker):
    return mocker.patch('ceam_public_health.population.base_population.generate_ceam_population')


@pytest.fixture(scope='function')
def age_bounds_mock(mocker):
    return mocker.patch('ceam_public_health.population.base_population._assign_demography_with_age_bounds')


@pytest.fixture(scope='function')
def initial_age_mock(mocker):
    return mocker.patch('ceam_public_health.population.base_population._assign_demography_with_initial_age')


@pytest.fixture(scope='function')
def get_pop_data_mock(mocker):
    return mocker.patch('ceam_public_health.population.base_population._get_population_data')


@pytest.fixture(scope='function')
def assign_proportions_mock(mocker):
    return mocker.patch('ceam_public_health.population.base_population.assign_demographic_proportions')


@pytest.fixture(scope='function')
def get_populations_mock(mocker):
    return mocker.patch('ceam_public_health.population.base_population.get_populations')


@pytest.fixture(scope='function')
def get_subregions_mock(mocker):
    return mocker.patch('ceam_public_health.population.base_population.get_subregions')


def make_base_simulants():
    simulant_ids = range(100000)
    creation_time = pd.Timestamp(1990, 7, 2)
    return pd.DataFrame({'entrance_time': pd.Series(pd.Timestamp(creation_time), index=simulant_ids),
                         'exit_time': pd.Series(pd.NaT, index=simulant_ids),
                         'alive': pd.Series('alive', index=simulant_ids).astype(
                             pd.api.types.CategoricalDtype(categories=['alive', 'dead', 'untracked'], ordered=False))},
                        index=simulant_ids)


def make_full_simulants():
    base_simulants = make_base_simulants()
    base_simulants['location'] = pd.Series(1, index=base_simulants.index)
    base_simulants['sex'] = pd.Series('Male', index=base_simulants.index).astype(
        pd.api.types.CategoricalDtype(categories=['Male', 'Female'], ordered=False))
    base_simulants['age'] = np.random.uniform(0, 100, len(base_simulants))
    return base_simulants


def make_uniform_pop_data():
    age_bins = [(n, n + 2.5, n + 5) for n in range(0, 100, 5)]
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
                        'location': locations,
                        'population': [100] * len(ages)})
    pop.loc[pop.sex == 'Both', 'population'] = 200
    return pop


def test_BasePopulation(config, build_pop_data_table_mock, generate_ceam_population_mock):
    num_days = 600
    time_step = 100  # Days
    uniform_pop = dt.assign_demographic_proportions(make_uniform_pop_data())
    sims = make_full_simulants()
    start_population_size = len(sims)

    build_pop_data_table_mock.return_value = uniform_pop
    generate_ceam_population_mock.return_value = sims

    use_subregions = ('use_subregions' in config.population and config.population.use_subregions)

    base_pop = bp.BasePopulation()

    components = [base_pop]
    simulation = setup_simulation(components, population_size=start_population_size, input_config=config)
    time_start = simulation.clock.time

    build_pop_data_table_mock.assert_called_once_with(config.input_data.location, use_subregions, config)
    assert base_pop._population_data.equals(uniform_pop)

    age_params = {'age_start': config.population.age_start,
                  'age_end': config.population.age_end}
    sub_pop = uniform_pop[uniform_pop.year == time_start.year]

    generate_ceam_population_mock.assert_called_once()
    # Get a dictionary of the arguments used in the call
    mock_args = generate_ceam_population_mock.call_args[1]
    assert mock_args['creation_time'] == time_start - simulation.clock.step_size
    assert mock_args['age_params'] == age_params
    assert mock_args['population_data'].equals(sub_pop)
    assert mock_args['randomness_streams'] == base_pop.randomness
    for column in simulation.population.population:
        assert simulation.population.population[column].equals(sims[column])

    final_ages = simulation.population.population.age + num_days/365

    pump_simulation(simulation, time_step_days=time_step, duration=pd.Timedelta(days=num_days))

    assert np.allclose(simulation.population.population.age, final_ages, atol=0.5/365)  # Within a half of a day.


def test_age_out_simulants(config):
    start_population_size = 10000
    num_days = 600
    time_step = 100  # Days
    config.update({'population': {'age_start': 4,
                                  'age_end': 4,
                                  'exit_age': 5, },
                   'time': {'step_size': time_step}
                   },
                  layer='override')
    components = [bp.BasePopulation()]
    simulation = setup_simulation(components, population_size=start_population_size, input_config=config)
    time_start = simulation.clock.time

    assert len(simulation.population.population) == len(simulation.population.population.age.unique())
    pump_simulation(simulation, time_step_days=time_step, duration=pd.Timedelta(days=num_days))
    pop = simulation.population.population
    assert len(pop) == len(pop[pop.alive == 'untracked'])
    exit_after_300_days = pop.exit_time >= time_start + pd.Timedelta(300, unit='D')
    exit_before_400_days = pop.exit_time <= time_start + pd.Timedelta(400, unit='D')
    assert len(pop) == len(pop[exit_after_300_days & exit_before_400_days])


def test_generate_ceam_population_age_bounds(age_bounds_mock, initial_age_mock):
    creation_time = pd.Timestamp(1990, 7, 2)
    step_size = pd.Timedelta(days=1)
    age_params = {'age_start': 0,
                  'age_end': 120}
    pop_data = dt.assign_demographic_proportions(make_uniform_pop_data())
    r = {k: get_randomness() for k in ['general_purpose', 'bin_selection', 'age_smoothing']}
    register = lambda *args, **kwargs: None
    sims = make_base_simulants()
    simulant_ids = sims.index

    bp.generate_ceam_population(simulant_ids, creation_time, step_size, age_params, pop_data, r, register)

    age_bounds_mock.assert_called_once()
    mock_args = age_bounds_mock.call_args[0]
    assert mock_args[0].equals(sims)
    assert mock_args[1].equals(pop_data)
    assert mock_args[2] == float(age_params['age_start'])
    assert mock_args[3] == float(age_params['age_end'])
    assert mock_args[4] == r
    initial_age_mock.assert_not_called()


def test_generate_ceam_population_initial_age(age_bounds_mock, initial_age_mock):
    creation_time = pd.Timestamp(1990, 7, 2)
    step_size = pd.Timedelta(days=1)
    age_params = {'age_start': 0,
                  'age_end': 0}
    pop_data = dt.assign_demographic_proportions(make_uniform_pop_data())
    r = {k: get_randomness() for k in ['general_purpose', 'bin_selection', 'age_smoothing']}
    register = lambda *args, **kwargs: None
    sims = make_base_simulants()
    simulant_ids = sims.index

    bp.generate_ceam_population(simulant_ids, creation_time, step_size, age_params, pop_data, r, register)

    initial_age_mock.assert_called_once()
    mock_args = initial_age_mock.call_args[0]
    assert mock_args[0].equals(sims)
    assert mock_args[1].equals(pop_data)

    assert mock_args[2] == float(age_params['age_start'])
    assert mock_args[3] == step_size
    assert mock_args[4] == r
    age_bounds_mock.assert_not_called()


def test__assign_demography_with_initial_age(config):
    pop_data = dt.assign_demographic_proportions(make_uniform_pop_data())
    pop_data = pop_data[pop_data.year == 1990]
    simulants = make_base_simulants()
    initial_age = 20
    r = {k: get_randomness() for k in ['general_purpose', 'bin_selection', 'age_smoothing']}
    register = lambda *args, **kwargs: None
    step_size = pd.Timedelta(days=config.time.step_size)

    simulants = bp._assign_demography_with_initial_age(simulants, pop_data, initial_age, step_size, r, register)

    assert len(simulants) == len(simulants.age.unique())
    assert simulants.age.min() > initial_age
    assert simulants.age.max() < initial_age + step_size.days/365.0
    assert math.isclose(len(simulants[simulants.sex == 'Male']) / len(simulants), 0.5, abs_tol=0.01)
    for location in simulants.location.unique():
        assert math.isclose(len(simulants[simulants.location == location]) / len(simulants),
                            1 / len(simulants.location.unique()), abs_tol=0.01)


def test__assign_demography_with_initial_age_zero(config):
    pop_data = dt.assign_demographic_proportions(make_uniform_pop_data())
    pop_data = pop_data[pop_data.year == 1990]
    simulants = make_base_simulants()
    initial_age = 0
    r = {k: get_randomness() for k in ['general_purpose', 'bin_selection', 'age_smoothing']}
    register = lambda *args, **kwargs: None
    step_size = pd.Timedelta(days=config.time.step_size)

    simulants = bp._assign_demography_with_initial_age(simulants, pop_data, initial_age, step_size, r, register)

    assert len(simulants) == len(simulants.age.unique())
    assert simulants.age.min() > initial_age
    assert simulants.age.max() < initial_age + step_size.days / 365.0
    assert math.isclose(len(simulants[simulants.sex == 'Male']) / len(simulants), 0.5, abs_tol=0.01)
    for location in simulants.location.unique():
        assert math.isclose(len(simulants[simulants.location == location]) / len(simulants),
                            1 / len(simulants.location.unique()), abs_tol=0.01)


def test__assign_demography_with_initial_age_error():
    pop_data = dt.assign_demographic_proportions(make_uniform_pop_data())
    pop_data = pop_data[pop_data.year == 1990]
    simulants = make_base_simulants()
    initial_age = 200
    r = {k: get_randomness() for k in ['general_purpose', 'bin_selection', 'age_smoothing']}
    register = lambda *args, **kwargs: None
    step_size = pd.Timedelta(days=1)

    with pytest.raises(ValueError):
        bp._assign_demography_with_initial_age(simulants, pop_data, initial_age, step_size, r, register)


def test__assign_demography_with_age_bounds():
    pop_data = dt.assign_demographic_proportions(make_uniform_pop_data())
    pop_data = pop_data[pop_data.year == 1990]
    simulants = make_base_simulants()
    age_start, age_end = 0, 180
    r = {k: get_randomness() for k in ['general_purpose', 'bin_selection', 'age_smoothing']}
    register = lambda *args, **kwargs: None

    simulants = bp._assign_demography_with_age_bounds(simulants, pop_data, age_start, age_end, r, register)

    assert math.isclose(len(simulants[simulants.sex == 'Male']) / len(simulants), 0.5, abs_tol=0.01)

    for location in simulants.location.unique():
        assert math.isclose(len(simulants[simulants.location == location]) / len(simulants),
                            1 / len(simulants.location.unique()), abs_tol=0.01)
    ages = np.sort(simulants.age.values)
    age_deltas = ages[1:] - ages[:-1]

    age_bin_width = 5  # See `make_uniform_pop_data`
    num_bins = len(pop_data.age.unique())
    n = len(simulants)
    assert math.isclose(age_deltas.mean(), age_bin_width * num_bins / n, rel_tol=1e-3)
    assert age_deltas.max() < 100 * age_bin_width * num_bins / n  # Make sure there are no big age gaps.


def test__assign_demography_withq_age_bounds_error():
    pop_data = dt.assign_demographic_proportions(make_uniform_pop_data())
    simulants = make_base_simulants()
    age_start, age_end = 110, 120
    r = {k: get_randomness() for k in ['general_purpose', 'bin_selection', 'age_smoothing']}
    register = lambda *args, **kwargs: None

    with pytest.raises(ValueError):
        bp._assign_demography_with_age_bounds(simulants, pop_data, age_start, age_end, r, register)


def test__build_population_data_table(config, get_pop_data_mock, assign_proportions_mock):
    df = pd.DataFrame({'A': np.arange(10), 'B': np.arange(10)})
    get_pop_data_mock.return_value = df
    assign_proportions_mock.return_value = 1
    test = bp._build_population_data_table(1, True, config)

    get_pop_data_mock.assert_called_once_with(1, True, config)
    assign_proportions_mock.assert_called_once_with(df)
    assert test == 1


def test__get_population_data(config, get_populations_mock, get_subregions_mock, mocker):

    main_id = 10
    main_id_no_subregions = 20
    subregion_ids = [11, 12]
    year_start = config.time.start.year
    year_end = config.time.end.year

    get_subregions_mock.side_effect = lambda override_config: (subregion_ids if override_config.input_data.location
                                                                                == main_id else None)
    test_populations = {
        10: build_table(20, year_start, year_end, ['age', 'year', 'sex', 'population']),
        11: build_table(30, year_start, year_end, ['age', 'year', 'sex', 'population']),
        12: build_table(50, year_start, year_end, ['age', 'year', 'sex', 'population']),
        20: build_table(70, year_start, year_end, ['age', 'year', 'sex', 'population']),
    }
    get_populations_mock.side_effect = lambda override_config, location: test_populations[location]

    config.input_data.location = main_id
    bp._get_population_data(main_id, True, config)
    get_subregions_mock.assert_called_once_with(config)
    assert get_populations_mock.call_args_list == [mocker.call(override_config=config, location=loc)
                                                   for loc in subregion_ids]

    get_subregions_mock.reset_mock()
    get_populations_mock.reset_mock()

    bp._get_population_data(main_id, False, config)
    get_subregions_mock.assert_not_called()
    get_populations_mock.assert_called_once_with(location=main_id, override_config=config)

    get_subregions_mock.reset_mock()
    get_populations_mock.reset_mock()

    config.input_data.location = main_id_no_subregions
    bp._get_population_data(main_id_no_subregions, True, config)
    get_subregions_mock.assert_called_once_with(config)
    get_populations_mock.assert_called_once_with(location=main_id_no_subregions, override_config=config)
