import os
import math
from itertools import product
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from vivarium import config
from vivarium.test_util import setup_simulation, pump_simulation, build_table, get_randomness

import ceam_public_health.population.base_population as bp
import ceam_public_health.population.data_transformations as dt


def setup():
    try:
        config.reset_layer('override', preserve_keys=['input_data.intermediary_data_cache_path',
                                                      'input_data.auxiliary_data_folder'])
    except KeyError:
        pass
    config.simulation_parameters.set_with_metadata('pop_age_start', 0, layer='override',
                                                   source=os.path.realpath(__file__))
    config.simulation_parameters.set_with_metadata('pop_age_end', 110, layer='override',
                                                   source=os.path.realpath(__file__))


def make_base_simulants():
    simulant_ids = range(100000)
    creation_time = datetime(1990, 7, 2)
    return pd.DataFrame({'entrance_time': pd.Series(pd.Timestamp(creation_time), index=simulant_ids),
                         'exit_time': pd.Series(pd.NaT, index=simulant_ids),
                         'alive': pd.Series('alive', index=simulant_ids).astype(
                             'category', categories=['alive', 'dead', 'untracked'], ordered=False)},
                        index=simulant_ids)


def make_full_simulants():
    base_simulants = make_base_simulants()
    base_simulants['location'] = pd.Series(1, index=base_simulants.index)
    base_simulants['sex'] = pd.Series('Male', index=base_simulants.index).astype(
        'category', categories=['Male', 'Female'], ordered=False)
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
                        'location_id': locations,
                        'pop_scaled': [100] * len(ages)})
    pop.loc[pop.sex == 'Both', 'pop_scaled'] = 200
    return pop


@patch('ceam_public_health.population.base_population.generate_ceam_population')
@patch('ceam_public_health.population.base_population._build_population_data_table')
def test_BasePopulation(build_pop_data_table_mock, generate_ceam_population_mock):
    num_days = 600
    time_step = 100  # Days
    time_start = pd.Timestamp('1990-01-01')
    uniform_pop = dt.assign_demographic_proportions(make_uniform_pop_data())
    sims = make_full_simulants()
    start_population_size = len(sims)

    build_pop_data_table_mock.return_value = uniform_pop
    generate_ceam_population_mock.return_value = sims

    base_pop = bp.BasePopulation()

    use_subregions = ('use_subregions' in config.simulation_parameters
                      and config.simulation_parameters.use_subregions)

    build_pop_data_table_mock.assert_called_once_with(config.simulation_parameters.location_id, use_subregions)
    assert base_pop._population_data.equals(uniform_pop)

    components = [base_pop]
    simulation = setup_simulation(components, population_size=start_population_size, start=time_start)

    with_initial_age = ('initial_age' in config.simulation_parameters
                        and config.simulation_paramters.initial_age is not None)
    initial_age = config.simulation_parameters.initial_age if with_initial_age else None
    age_params = {'initial_age': initial_age,
                  'pop_age_start': config.simulation_parameters.pop_age_start,
                  'pop_age_end': config.simulation_parameters.pop_age_start}
    sub_pop = uniform_pop[uniform_pop.year == time_start.year]

    generate_ceam_population_mock.assert_called_once()
    # Get a dictionary of the arguments used in the call
    mock_args = generate_ceam_population_mock.call_args[1]
    assert mock_args['creation_time'] == time_start
    assert mock_args['age_params'] == age_params
    assert mock_args['population_data'].equals(sub_pop)
    assert mock_args['randomness_stream'] == base_pop.randomness
    for column in simulation.population.population:
        assert simulation.population.population[column].equals(sims[column])

    final_ages = simulation.population.population.age + num_days/365

    pump_simulation(simulation, time_step_days=time_step, duration=pd.Timedelta(days=num_days))

    assert np.allclose(simulation.population.population.age, final_ages)


def test_age_out_simulants():
    start_population_size = 10000
    num_days = 600
    time_step = 100  # Days
    time_start = pd.Timestamp('1990-01-01')
    config.read_dict({'simulation_parameters': {'initial_age': 4,
                                                'maximum_age': 5,
                                                'time_step': time_step}},
                     layer='override')
    components = [bp.BasePopulation(), bp.age_out_simulants]
    simulation = setup_simulation(components, population_size=start_population_size, start=time_start)
    pump_simulation(simulation, time_step_days=time_step, duration=pd.Timedelta(days=num_days))
    pop = simulation.population.population
    assert len(pop) == len(pop[pop.alive == 'untracked'])
    assert len(pop) == len(pop[pop.exit_time == time_start + pd.Timedelta(300, unit='D')])


@patch('ceam_public_health.population.base_population._assign_demography_with_initial_age')
@patch('ceam_public_health.population.base_population._assign_demography_with_age_bounds')
def test_generate_ceam_population_age_bounds(age_bounds_mock, initial_age_mock):
    creation_time = datetime(1990, 7, 2)
    age_params = {'initial_age': None,
                  'pop_age_start': 0,
                  'pop_age_end': 120}
    pop_data = dt.assign_demographic_proportions(make_uniform_pop_data())
    r = get_randomness()
    sims = make_base_simulants()
    simulant_ids = sims.index

    bp.generate_ceam_population(simulant_ids, creation_time, age_params, pop_data, r)

    age_bounds_mock.assert_called_once()
    mock_args = age_bounds_mock.call_args[0]
    assert mock_args[0].equals(sims)
    assert mock_args[1].equals(pop_data)
    assert mock_args[2] == float(age_params['pop_age_start'])
    assert mock_args[3] == float(age_params['pop_age_end'])
    assert mock_args[4] == r
    initial_age_mock.assert_not_called()


@patch('ceam_public_health.population.base_population._assign_demography_with_initial_age')
@patch('ceam_public_health.population.base_population._assign_demography_with_age_bounds')
def test_generate_ceam_population_initial_age(age_bounds_mock, initial_age_mock):
    creation_time = datetime(1990, 7, 2)
    age_params = {'initial_age': 0,
                  'pop_age_start': 0,
                  'pop_age_end': 120}
    pop_data = dt.assign_demographic_proportions(make_uniform_pop_data())
    r = get_randomness()
    sims = make_base_simulants()
    simulant_ids = sims.index

    bp.generate_ceam_population(simulant_ids, creation_time, age_params, pop_data, r)

    initial_age_mock.assert_called_once()
    mock_args = initial_age_mock.call_args[0]
    assert mock_args[0].equals(sims)
    assert mock_args[1].equals(pop_data)
    assert mock_args[2] == float(age_params['initial_age'])
    assert mock_args[3] == r
    age_bounds_mock.assert_not_called()


def test__assign_demography_with_initial_age():
    pop_data = dt.assign_demographic_proportions(make_uniform_pop_data())
    pop_data = pop_data[pop_data.year == 1990]
    simulants = make_base_simulants()
    initial_age = 20
    r = get_randomness()

    simulants = bp._assign_demography_with_initial_age(simulants, pop_data, initial_age, r)

    assert np.all(simulants.age == initial_age)
    assert math.isclose(len(simulants[simulants.sex == 'Male']) / len(simulants), 0.5, abs_tol=0.01)
    for location in simulants.location.unique():
        assert math.isclose(len(simulants[simulants.location == location]) / len(simulants),
                            1 / len(simulants.location.unique()), abs_tol=0.01)


def test__assign_demography_with_initial_age_zero():
    pop_data = dt.assign_demographic_proportions(make_uniform_pop_data())
    pop_data = pop_data[pop_data.year == 1990]
    simulants = make_base_simulants()
    initial_age = 0
    r = get_randomness()

    simulants = bp._assign_demography_with_initial_age(simulants, pop_data, initial_age, r)

    assert not simulants.age.values.any()
    assert math.isclose(len(simulants[simulants.sex == 'Male']) / len(simulants), 0.5, abs_tol=0.01)
    for location in simulants.location.unique():
        assert math.isclose(len(simulants[simulants.location == location]) / len(simulants),
                            1 / len(simulants.location.unique()), abs_tol=0.01)


def test__assign_demography_with_initial_age_error():
    pop_data = dt.assign_demographic_proportions(make_uniform_pop_data())
    pop_data = pop_data[pop_data.year == 1990]
    simulants = make_base_simulants()
    initial_age = 200
    r = get_randomness()

    with pytest.raises(ValueError):
        bp._assign_demography_with_initial_age(simulants, pop_data, initial_age, r)


def test__assign_demography_with_age_bounds():
    pop_data = dt.assign_demographic_proportions(make_uniform_pop_data())
    pop_data = pop_data[pop_data.year == 1990]
    simulants = make_base_simulants()
    age_start, age_end = 0, 180
    r = get_randomness()

    simulants = bp._assign_demography_with_age_bounds(simulants, pop_data, age_start, age_end, r)

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


def test__assign_demography_with_age_bounds_error():
    pop_data = dt.assign_demographic_proportions(make_uniform_pop_data())
    simulants = make_base_simulants()
    age_start, age_end = 110, 120
    r = get_randomness()

    with pytest.raises(ValueError):
        bp._assign_demography_with_age_bounds(simulants, pop_data, age_start, age_end, r)


@patch('ceam_public_health.population.base_population._get_population_data')
@patch('ceam_public_health.population.base_population.assign_demographic_proportions')
def test__build_population_data_table(assign_proportions_mock, get_pop_data_mock):
    df = pd.DataFrame({'A': np.arange(10), 'B': np.arange(10)})
    get_pop_data_mock.return_value = df
    assign_proportions_mock.return_value = 1
    test = bp._build_population_data_table(1, True)

    get_pop_data_mock.assert_called_once_with(1, True)
    assign_proportions_mock.assert_called_once_with(df)
    assert test == 1


@patch('ceam_public_health.population.base_population.get_populations')
@patch('ceam_public_health.population.base_population.get_subregions')
def test__get_population_data(get_subregions_mock, get_populations_mock):
    main_id = 10
    main_id_no_subregions = 20
    subregion_ids = [11, 12]

    get_subregions_mock.side_effect = lambda location_id: subregion_ids if location_id == main_id else None
    test_populations = {
        10: build_table(20, ['age', 'year', 'sex', 'pop_scaled']),
        11: build_table(30, ['age', 'year', 'sex', 'pop_scaled']),
        12: build_table(50, ['age', 'year', 'sex', 'pop_scaled']),
        20: build_table(70, ['age', 'year', 'sex', 'pop_scaled']),
    }
    get_populations_mock.side_effect = lambda location_id: test_populations[location_id]

    bp._get_population_data(main_id, True)
    get_subregions_mock.assert_called_once_with(main_id)
    assert get_populations_mock.call_args_list == [({'location_id': loc},) for loc in subregion_ids]

    get_subregions_mock.reset_mock()
    get_populations_mock.reset_mock()

    bp._get_population_data(main_id, False)
    get_subregions_mock.assert_not_called()
    get_populations_mock.assert_called_once_with(location_id=main_id)

    get_subregions_mock.reset_mock()
    get_populations_mock.reset_mock()

    bp._get_population_data(main_id_no_subregions, True)
    get_subregions_mock.assert_called_once_with(main_id_no_subregions)
    get_populations_mock.assert_called_once_with(location_id=main_id_no_subregions)
