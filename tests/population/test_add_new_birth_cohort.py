from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from vivarium import InteractiveContext
from vivarium.testing_utilities import TestPopulation, metadata, build_table

from vivarium_public_health import utilities
from vivarium_public_health.population import FertilityDeterministic, FertilityCrudeBirthRate, FertilityAgeSpecificRates


@pytest.fixture()
def config(base_config):

    base_config.update({
        'population': {
            'population_size': 10000,
            'age_start': 0,
            'age_end': 125,
        },
        'time': {
            'step_size': 10,
            }
        }, source=str(Path(__file__).resolve()), layer='override')
    return base_config


def crude_birth_rate_data(live_births=500):
    return (build_table(['mean_value', live_births], 1990, 2017, ('age', 'year', 'sex', 'parameter', 'value'))
            .query('age_start == 25 and sex != "Both"')
            .drop(['age_start', 'age_end'], 'columns'))


def test_FertilityDeterministic(config):
    pop_size = config.population.population_size
    annual_new_simulants = 1000
    step_size = config.time.step_size
    num_days = 100

    config.update({
        'fertility': {
            'number_of_new_simulants_each_year': annual_new_simulants
        }
    }, **metadata(__file__))

    components = [TestPopulation(), FertilityDeterministic()]
    simulation = InteractiveContext(components=components, configuration=config)
    num_steps = simulation.run_for(duration=pd.Timedelta(days=num_days))
    pop = simulation.get_population()

    assert num_steps == num_days // step_size
    assert np.all(pop.alive == 'alive')
    assert int(num_days * annual_new_simulants / utilities.DAYS_PER_YEAR) == len(pop.age) - pop_size


def test_FertilityCrudeBirthRate(config, base_plugins):
    pop_size = config.population.population_size
    num_days = 100
    components = [TestPopulation(), FertilityCrudeBirthRate()]
    simulation = InteractiveContext(components=components,
                                    configuration=config,
                                    plugin_configuration=base_plugins,
                                    setup=False)
    simulation._data.write("covariate.live_births_by_sex.estimate", crude_birth_rate_data())

    simulation.setup()
    simulation.run_for(duration=pd.Timedelta(days=num_days))
    pop = simulation.get_population()

    assert np.all(pop.alive == 'alive')
    assert len(pop.age) > pop_size


def test_FertilityCrudeBirthRate_extrapolate_fail(config, base_plugins):
    config.update({
        'interpolation': {
            'extrapolate': False
        },
        'time': {
            'start': {'year': 2016},
            'end': {'year': 2025},
        },
    })
    components = [TestPopulation(), FertilityCrudeBirthRate()]

    simulation = InteractiveContext(components=components,
                                    configuration=config,
                                    plugin_configuration=base_plugins,
                                    setup=False)
    simulation._data.write("covariate.live_births_by_sex.estimate", crude_birth_rate_data())

    with pytest.raises(ValueError):
        simulation.setup()


def test_FertilityCrudeBirthRate_extrapolate(base_config, base_plugins):
    base_config.update({
        'population': {
            'population_size': 10000,
            'age_start': 0,
            'age_end': 125,
        },
        'interpolation': {
            'extrapolate': True
        },
        'time': {
            'start': {'year': 2016},
            'end': {'year': 2026},
            'step_size': 365,
        },
    })
    pop_size = base_config.population.population_size
    true_pop_size = 8000  # What's available in the mock artifact
    live_births_by_sex = 500
    components = [TestPopulation(), FertilityCrudeBirthRate()]

    simulation = simulation = InteractiveContext(components=components,
                                                 configuration=base_config,
                                                 plugin_configuration=base_plugins,
                                                 setup=False)
    simulation._data.write("covariate.live_births_by_sex.estimate", crude_birth_rate_data(live_births_by_sex))
    simulation.setup()

    birth_rate = []
    for i in range(10):
        pop_start = len(simulation.get_population())
        simulation.step()
        pop_end = len(simulation.get_population())
        birth_rate.append((pop_end - pop_start)/pop_size)

    given_birth_rate = 2*live_births_by_sex / true_pop_size
    np.testing.assert_allclose(birth_rate, given_birth_rate, atol=0.01)


def test_fertility_module(base_config, base_plugins):
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
    simulation = simulation = InteractiveContext(components=components,
                                                 configuration=base_config,
                                                 plugin_configuration=base_plugins,
                                                 setup=False)

    asfr_data = build_table(0.05, 1990, 2017).rename(columns={'value': 'mean_value'})
    simulation._data.write("covariate.age_specific_fertility_rate.estimate", asfr_data)

    simulation.setup()

    time_start = simulation._clock.time

    assert 'last_birth_time' in simulation.get_population().columns,\
        'expect Fertility module to update state table.'
    assert 'parent_id' in simulation.get_population().columns, \
        'expect Fertility module to update state table.'

    simulation.run_for(duration=pd.Timedelta(days=num_days))
    pop = simulation.get_population()

    # No death in this model.
    assert np.all(pop.alive == 'alive'), 'expect all simulants to be alive'

    # TODO: Write a more rigorous test.
    assert len(pop.age) > start_population_size, 'expect new simulants'

    for i in range(start_population_size, len(pop)):
        assert pop.loc[pop.iloc[i].parent_id].last_birth_time >= time_start, 'expect all children to have mothers who' \
                                                                          ' gave birth after the simulation starts.'
