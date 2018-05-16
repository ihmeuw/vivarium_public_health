import os

import pytest
import numpy as np
import pandas as pd

from vivarium.test_util import setup_simulation, pump_simulation, TestPopulation, build_table

from ceam_public_health.dataset_manager import ArtifactManager
from ceam_public_health.testing.mock_artifact import MockArtifact
from ceam_public_health.population import FertilityDeterministic, FertilityCrudeBirthRate, FertilityAgeSpecificRates


@pytest.fixture(scope='function')
def config(base_config):
    try:
        base_config.reset_layer('override', preserve_keys=['input_data.intermediary_data_cache_path',
                                                           'input_data.auxiliary_data_folder'])
    except KeyError:
        pass

    metadata = {'layer': 'override', 'source': os.path.realpath(__file__)}
    base_config.time.start.set_with_metadata('year', 1990, **metadata)
    base_config.time.end.set_with_metadata('year', 2010, **metadata)
    base_config.time.set_with_metadata('step_size', 30.5, **metadata)
    base_config.input_data.set_with_metadata('location', 'Kenya', **metadata)
    base_config.vivarium.set_with_metadata('dataset_manager', 'ceam_public_health.dataset_manager.ArtifactManager', **metadata)
    base_config.input_data.set_with_metadata('artifact_path', '/tmp/dummy.hdf', **metadata)
    base_config.artifact.set_with_metadata('artifact_class', 'ceam_public_health.testing.mock_artifact.MockArtifact', **metadata)
    return base_config


def test_FertilityDeterministic(config):
    start_population_size = 1000
    annual_new_simulants = 1000
    num_days = 100
    time_step = 10  # Days
    config.read_dict({'population': {'age_start': 0, 'age_end': 125},
                      'fertility_deterministic': {'number_of_new_simulants_each_year': annual_new_simulants},
                      'time': {'step_size': time_step}},
                     layer='override')

    components = [TestPopulation(), FertilityDeterministic()]
    simulation = setup_simulation(components, population_size=start_population_size, input_config=config)
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
    config.read_dict({'population': {'age_start': 0, 'age_end': 125}}, layer='override')

    artifact = MockArtifact()
    artifact.set("covariate.age_specific_fertility_rate.estimate", 0.01)
    artifact.set("covariate.live_births_by_sex.estimate", build_table(5000, 1990, 2018, ('age', 'year', 'sex', 'mean_value')).query('age == 25').drop('age', 'columns'))
    components = [TestPopulation(), FertilityCrudeBirthRate()]
    simulation = setup_simulation(components, population_size=start_population_size, input_config=config, dataset_manager=ArtifactManager(artifact))

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
    config.read_dict({'population': {'age_start': 0, 'age_end': 125}}, layer='override')

    artifact = MockArtifact()
    artifact.set("covariate.age_specific_fertility_rate.estimate", build_table(0.05, 1990, 2018).query("sex == 'Female'").drop("sex", "columns"))

    components = [TestPopulation(), FertilityAgeSpecificRates()]
    simulation = setup_simulation(components, population_size=start_population_size, input_config=config, dataset_manager=ArtifactManager(artifact))
    time_start = simulation.clock.time

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
