import os

import pytest
import numpy as np
import pandas as pd

from vivarium.framework.state_machine import State
from vivarium.test_util import setup_simulation, build_table, TestPopulation, from_yearly

from ceam_public_health.disease import RateTransition, DiseaseModel


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


def make_model(config, incidence_rate, recovery_rate):
    year_start = config.simulation_parameters.year_start
    year_end = config.simulation_parameters.year_end
    healthy = State('healthy')
    sick = State('sick')
    recovered = State('recovered')

    healthy.transition_set.append(RateTransition(sick, 'infection',
                                                 build_table(incidence_rate, year_start, year_end)))
    sick.transition_set.append(RateTransition(healthy, 'recovery',
                                              build_table(recovery_rate, year_start, year_end)))

    model = DiseaseModel('simple_disease')
    model.states.extend([healthy, sick, recovered])

    return model


def test_incidence_rate_recalculation(config):
    config.simulation_parameters.set_with_metadata('time_step', 1, layer='override',
                                                   source=os.path.realpath(__file__))
    incidence_rate = 0.01
    recovery_rate = 72  # Average duration of 5 days
    sim = setup_simulation([TestPopulation(), make_model(config, incidence_rate, recovery_rate)],
                           population_size=50000, input_config=config)

    susceptible = [(sim.population.population.simple_disease == 'healthy').sum()]
    new_cases = []
    known_cases = pd.Index([])

    for step in range(360):
        sim.step()
        cases = sim.population.population.query('simple_disease == "sick"')
        new_cases.append(len(cases.index.difference(known_cases)))
        known_cases = cases.index
        susceptible.append((sim.population.population.simple_disease == 'healthy').sum())

    susceptible = susceptible[:-1]
    incidence_rates = np.array(new_cases)/np.array(susceptible)

    assert np.isclose(np.mean(incidence_rates), from_yearly(0.01, pd.Timedelta(days=1)), rtol=0.05)

    expected_prevalence = incidence_rate * (1/recovery_rate)
    actual_prevalence = (sim.population.population.simple_disease == 'sick').sum() / len(sim.population.population)

    assert np.isclose(expected_prevalence, actual_prevalence)
