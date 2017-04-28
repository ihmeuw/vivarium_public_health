# ~/ceam/ceam_tests/test_modules/test_add_new_birth_cohort.py


import pandas as pd
import numpy as np
from ceam_tests.util import setup_simulation, pump_simulation, generate_test_population
from ceam_public_health.components import add_new_birth_cohorts as anbc
from ceam import config


def test_add_new_birth_cohorts_deterministic():
    start_population_size = 1000
    annual_new_simulants = 1000
    num_days = 100
    time_step = 10  # Days
    time_start = pd.Timestamp('1990-01-01')
    config.read_dict({'simulation_parameters':
                          {'pop_age_start': 0, 'pop_age_end': 125,
                           'number_of_new_simulants_each_year': annual_new_simulants}},
                     layer='override')

    components = [generate_test_population, anbc.add_new_birth_cohort]
    simulation = setup_simulation(components, population_size=start_population_size, start=time_start)
    pump_simulation(simulation, time_step_days=time_step, duration=pd.Timedelta(days=num_days))
    pop = simulation.population.population

    # No death in this model.
    assert np.all(simulation.population.population.alive), 'expect all simulants to be alive'

    # We expect to have n_days/time_step steps each producing
    # ceil(1000 * time_step / 365) new simulants, so
    assert (num_days / time_step * np.ceil(annual_new_simulants * time_step / 365.)
            == len(pop.age) - start_population_size), 'expect new simulants'


def test_add_new_birth_cohorts_nondeterministic():
    start_population_size = 10000
    num_days = 100
    time_step = 10  # Days
    time_start = pd.Timestamp('1990-01-01')
    config.read_dict({'simulation_parameters': {'pop_age_start': 0, 'pop_age_end': 125}}, layer='override')

    components = [generate_test_population, anbc.add_new_birth_cohort_nondeterministic]
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

    components = [generate_test_population, anbc.Fertility()]
    simulation = setup_simulation(components, population_size=start_population_size, start=time_start)

    assert 'last_birth_time' in simulation.population.population.columns,\
        'expect Fertility module to update state table.'
    assert 'parent' in simulation.population.population.columns, \
        'expect Fertility module to update state table.'

    pump_simulation(simulation, time_step_days=time_step, duration=pd.Timedelta(days=num_days))
    pop = simulation.population.population

    # No death in this model.
    assert np.all(simulation.population.population.alive), 'expect all simulants to be alive'

    # TODO: Write a more rigorous test.
    assert len(pop.age) > start_population_size, 'expect new simulants'

    for i in range(start_population_size, len(pop)):
        assert pop.iloc[pop.iloc[i].parent].last_birth_time >= time_start, 'expect all children to have mothers who' \
                                                                          ' gave birth after the simulation starts.'
