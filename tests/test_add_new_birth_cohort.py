import pandas as pd
import numpy as np
from ceam_tests.util import setup_simulation, pump_simulation, generate_test_population
from ceam_public_health.components import add_new_birth_cohorts as anbc
from ceam import config


def test_FertilityDeterministic():
    start_population_size = 1000
    annual_new_simulants = 1000
    num_days = 100
    time_step = 10  # Days
    time_start = pd.Timestamp('1990-01-01')
    config.read_dict({'simulation_parameters':
                          {'pop_age_start': 0, 'pop_age_end': 125,
                           'number_of_new_simulants_each_year': annual_new_simulants,
                           'time_step': time_step}},
                     layer='override')

    components = [generate_test_population, anbc.FertilityDeterministic()]
    simulation = setup_simulation(components, population_size=start_population_size, start=time_start)
    num_steps = pump_simulation(simulation, time_step_days=time_step, duration=pd.Timedelta(days=num_days))
    assert num_steps == num_days // time_step
    pop = simulation.population.population

    # No death in this model.
    assert np.all(simulation.population.population.alive), 'expect all simulants to be alive'
    assert (int(num_days * annual_new_simulants / anbc.DAYS_PER_YEAR)
            == len(pop.age) - start_population_size), 'expect new simulants'


def test_FertilityCrudeBirthRate():
    start_population_size = 10000
    num_days = 100
    time_step = 10  # Days
    time_start = pd.Timestamp('1990-01-01')
    config.read_dict({'simulation_parameters': {'pop_age_start': 0, 'pop_age_end': 125}}, layer='override')

    components = [generate_test_population, anbc.FertilityCrudeBirthRate()]
    simulation = setup_simulation(components, population_size=start_population_size, start=time_start)
    pump_simulation(simulation, time_step_days=time_step, duration=pd.Timedelta(days=num_days))
    pop = simulation.population.population

    # No death in this model.
    assert np.all(simulation.population.population.alive), 'expect all simulants to be alive'

    # TODO: Write a more rigorous test.
    assert len(pop.age) > start_population_size, 'expect new simulants'


def test_fertility_module():
    start_population_size = 1000
    num_days = 1000
    time_step = 10  # Days
    time_start = pd.Timestamp('1990-01-01')
    config.read_dict({'simulation_parameters': {'pop_age_start': 0, 'pop_age_end': 125}}, layer='override')

    components = [generate_test_population, anbc.FertilityAgeSpecificRates()]
    simulation = setup_simulation(components, population_size=start_population_size, start=time_start)

    assert 'last_birth_time' in simulation.population.population.columns,\
        'expect Fertility module to update state table.'
    assert 'parent_id' in simulation.population.population.columns, \
        'expect Fertility module to update state table.'

    pump_simulation(simulation, time_step_days=time_step, duration=pd.Timedelta(days=num_days))
    pop = simulation.population.population

    # No death in this model.
    assert np.all(simulation.population.population.alive), 'expect all simulants to be alive'

    # TODO: Write a more rigorous test.
    assert len(pop.age) > start_population_size, 'expect new simulants'

    for i in range(start_population_size, len(pop)):
        assert pop.iloc[pop.iloc[i].parent_id].last_birth_time >= time_start, 'expect all children to have mothers who' \
                                                                          ' gave birth after the simulation starts.'
