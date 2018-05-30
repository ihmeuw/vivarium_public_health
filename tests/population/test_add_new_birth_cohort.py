import numpy as np
import pandas as pd

from vivarium.test_util import TestPopulation, metadata
from vivarium.interface.interactive import setup_simulation

from ceam_public_health.population import FertilityDeterministic, FertilityCrudeBirthRate, FertilityAgeSpecificRates


def test_FertilityDeterministic(base_config):
    start_population_size = 1000
    annual_new_simulants = 1000
    num_days = 100
    time_step = 10  # Days
    base_config.update({
        'population': {
            'population_size': start_population_size,
            'age_start': 0,
            'age_end': 125
        },
        'fertility_deterministic': {'number_of_new_simulants_each_year': annual_new_simulants},
        'time': {'step_size': time_step}
    }, **metadata(__file__))

    components = [TestPopulation(), FertilityDeterministic()]
    simulation = setup_simulation(components, base_config)
    num_steps = simulation.run_for(duration=pd.Timedelta(days=num_days))
    assert num_steps == num_days // time_step
    pop = simulation.population.population

    # No death in this model.
    assert np.all(simulation.population.population.alive == 'alive'), 'expect all simulants to be alive'
    assert (int(num_days * annual_new_simulants / 365)
            == len(pop.age) - start_population_size), 'expect new simulants'


def test_FertilityCrudeBirthRate(base_config):
    start_population_size = 10000
    num_days = 100
    time_step = 10  # Days
    base_config.update({
        'population': {
            'population_size': start_population_size,
            'age_start': 0,
            'age_end': 125},
        'time': {'step_size': time_step}
    }, **metadata(__file__))

    components = [TestPopulation(), FertilityCrudeBirthRate()]
    simulation = setup_simulation(components, base_config)
    simulation.run_for(duration=pd.Timedelta(days=num_days))
    pop = simulation.population.population

    # No death in this model.
    assert np.all(simulation.population.population.alive == 'alive'), 'expect all simulants to be alive'

    # TODO: Write a more rigorous test.
    assert len(pop.age) > start_population_size, 'expect new simulants'


def test_fertility_module(base_config):
    start_population_size = 1000
    num_days = 1000
    time_step = 10  # Days
    base_config.update({
        'population': {
            'population_size': start_population_size,
            'age_start': 0,
            'age_end': 125},
        'time': {'step_size': time_step}
    }, layer='override')

    components = [TestPopulation(), FertilityAgeSpecificRates()]
    simulation = setup_simulation(components, base_config)
    time_start = simulation.clock.time

    assert 'last_birth_time' in simulation.population.population.columns,\
        'expect Fertility module to update state table.'
    assert 'parent_id' in simulation.population.population.columns, \
        'expect Fertility module to update state table.'

    simulation.run_for(duration=pd.Timedelta(days=num_days))
    pop = simulation.population.population

    # No death in this model.
    assert np.all(simulation.population.population.alive == 'alive'), 'expect all simulants to be alive'

    # TODO: Write a more rigorous test.
    assert len(pop.age) > start_population_size, 'expect new simulants'

    for i in range(start_population_size, len(pop)):
        assert pop.loc[pop.iloc[i].parent_id].last_birth_time >= time_start, 'expect all children to have mothers who' \
                                                                          ' gave birth after the simulation starts.'
