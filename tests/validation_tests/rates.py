import os

from datetime import timedelta

import pandas as pd
import numpy as np

from ceam import config
from ceam.framework.engine import _step
from ceam.framework.state_machine import State
from ceam_tests.util import setup_simulation, build_table, generate_test_population

from ceam_public_health.components.disease import DiseaseModel, RateTransition


def setup():
    try:
        config.reset_layer('override', preserve_keys=['input_data.intermediary_data_cache_path',
                                                      'input_data.auxiliary_data_folder'])
    except KeyError:
        pass
    config.simulation_parameters.set_with_metadata('year_start', 1990, layer='override',
                                                   source=os.path.realpath(__file__))
    config.simulation_parameters.set_with_metadata('year_end', 2010, layer='override',
                                                   source=os.path.realpath(__file__))
    config.simulation_parameters.set_with_metadata('time_step', 30.5, layer='override',
                                                   source=os.path.realpath(__file__))

def make_model(incidence_rate, recovery_rate):
    healthy = State('healthy')
    sick = State('sick')
    recovered = State('recovered')

    healthy.transition_set.append(RateTransition(sick, 'infection', build_table(incidence_rate)))
    sick.transition_set.append(RateTransition(healthy, 'recovery', build_table(recovery_rate)))

    model = DiseaseModel('simple_disease')
    model.states.extend([healthy, sick, recovered])

    return model

def test_incidence_rate_recalculation():
    config.simulation_parameters.set_with_metadata('time_step', 1, layer='override',
                                                   source=os.path.realpath(__file__))
    incidence_rate = 0.01
    recovery_rate = 72 # Average duration of 5 days
    sim = setup_simulation([generate_test_population, make_model(incidence_rate, recovery_rate)], population_size=50000)

    susceptible = [(sim.population.population.simple_disease == 'healthy').sum()]
    new_cases = []
    known_cases = pd.Index([])

    for step in range(360):
        _step(sim, timedelta(days=1))
        cases = sim.population.population.query('simple_disease == "sick"')
        new_cases.append(len(cases.index.difference(known_cases)))
        known_cases = cases.index
        susceptible.append((sim.population.population.simple_disease == 'healthy').sum())

    susceptible = susceptible[:-1]
    incidence_rates = np.array(new_cases)/np.array(susceptible)

    #assert np.isclose(np.mean(incidence_rates), from_yearly(0.01, timedelta(days=1)), rtol=0.05)

    expected_prevalence = incidence_rate * (1/recovery_rate)
    actual_prevalence = (sim.population.population.simple_disease == 'sick').sum() / len(sim.population.population)

    #assert np.isclose(expected_prevalence, actual_prevalence)
