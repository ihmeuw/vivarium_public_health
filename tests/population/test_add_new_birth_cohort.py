import os

import pytest
import numpy as np
import pandas as pd

from vivarium.test_util import setup_simulation, pump_simulation, TestPopulation

from ceam_public_health.population import FertilityDeterministic, FertilityCrudeBirthRate, FertilityAgeSpecificRates


@pytest.fixture(scope='function')
def config(base_config):
    try:
        base_config.reset_layer('override', preserve_keys=['input_data.intermediary_data_cache_path',
                                                           'input_data.auxiliary_data_folder'])
    except KeyError:
        pass

    metadata = {'layer': 'override', 'source': os.path.realpath(__file__)}
    base_config.simulation_parameters.set_with_metadata('year_start', 1990, **metadata)
    base_config.simulation_parameters.set_with_metadata('year_end', 2010, **metadata)
    base_config.simulation_parameters.set_with_metadata('time_step', 30.5, **metadata)
    return base_config


def test_FertilityDeterministic(config):
    start_population_size = 1000
    annual_new_simulants = 1000
    num_days = 100
    time_step = 10  # Days
    time_start = pd.Timestamp('1990-01-01')
    config.read_dict({'population': {'pop_age_start': 0, 'pop_age_end': 125},
                      'fertility_deterministic': {'number_of_new_simulants_each_year': annual_new_simulants},
                      'simulation_parameters': {'time_step': time_step}},
                     layer='override')

    components = [TestPopulation(), FertilityDeterministic()]
    simulation = setup_simulation(components, population_size=start_population_size,
                                  start=time_start, input_config=config)
    num_steps = pump_simulation(simulation, time_step_days=time_step, duration=pd.Timedelta(days=num_days))
    assert num_steps == num_days // time_step
    pop = simulation.population.population

    # No death in this model.
    assert np.all(simulation.population.population.alive == 'alive'), 'expect all simulants to be alive'
    assert (int(num_days * annual_new_simulants / 365)
            == len(pop.age) - start_population_size), 'expect new simulants'


def test_FertilityCrudeBirthRate(config):
    start_population_size = 10000
    num_days = 100
    time_step = 10  # Days
    time_start = pd.Timestamp('1990-01-01')
    config.read_dict({'population': {'pop_age_start': 0, 'pop_age_end': 125}}, layer='override')

    components = [TestPopulation(), FertilityCrudeBirthRate()]
    simulation = setup_simulation(components, population_size=start_population_size,
                                  start=time_start, input_config=config)
    pump_simulation(simulation, time_step_days=time_step, duration=pd.Timedelta(days=num_days))
    pop = simulation.population.population

    # No death in this model.
    assert np.all(simulation.population.population.alive == 'alive'), 'expect all simulants to be alive'

    # TODO: Write a more rigorous test.
    assert len(pop.age) > start_population_size, 'expect new simulants'


def test_fertility_module(config):
    start_population_size = 1000
    num_days = 1000
    time_step = 10  # Days
    time_start = pd.Timestamp('1990-01-01')
    config.read_dict({'population': {'pop_age_start': 0, 'pop_age_end': 125}}, layer='override')

    components = [TestPopulation(), FertilityAgeSpecificRates()]
    simulation = setup_simulation(components, population_size=start_population_size,
                                  start=time_start, input_config=config)

    assert 'last_birth_time' in simulation.population.population.columns,\
        'expect Fertility module to update state table.'
    assert 'parent_id' in simulation.population.population.columns, \
        'expect Fertility module to update state table.'

    pump_simulation(simulation, time_step_days=time_step, duration=pd.Timedelta(days=num_days))
    pop = simulation.population.population

    # No death in this model.
    assert np.all(simulation.population.population.alive == 'alive'), 'expect all simulants to be alive'

    # TODO: Write a more rigorous test.
    assert len(pop.age) > start_population_size, 'expect new simulants'

    for i in range(start_population_size, len(pop)):
        assert pop.loc[pop.iloc[i].parent_id].last_birth_time >= time_start, 'expect all children to have mothers who' \
                                                                          ' gave birth after the simulation starts.'
