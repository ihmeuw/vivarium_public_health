# ~/ceam/ceam_tests/test_modules/test_disease.py

import pytest
from unittest.mock import Mock, patch
from datetime import timedelta

import pandas as pd
import numpy as np

from ceam import config

from ceam_tests.util import setup_simulation, pump_simulation, build_table

from ceam.framework.util import from_yearly

from ceam_inputs import get_incidence

from ceam_public_health.components.base_population import generate_base_population

from ceam.framework.state_machine import Transition, State
from ceam.framework.event import Event
from ceam_public_health.components.disease_models_dir.diarrhea_disease_model import ApplyDiarrheaExcessMortality, ApplyDiarrheaRemission

def test_ApplyDiarrheaRemission():
    # give everyone diarrhea, set remission to one time step for one age group, two timesteps for the other age group, pump the simulation twice to make sure remission does what it's expected to

def test__move_people_into_diarrhea_state():
    # give only people in a certain age group diarrhea due to a bunch of different pathogens and make sure they all end up in the same spot

# def test 

def test_mortality_rate():
    time_step = config.getfloat('simulation_parameters', 'time_step')
    time_step = timedelta(days=time_step)

    model = DiseaseModel('test_disease')
    healthy = State('healthy')
    mortality_state = ExcessMortalityState('sick', excess_mortality_data=build_table(0.7), disability_weight=0.1, prevalence_data=build_table(0.0, ['age', 'year', 'sex', 'prevalence']), csmr_data=build_table(0.0))

    healthy.transition_set.append(Transition(mortality_state))

    model.states.extend([healthy, mortality_state])

    simulation = setup_simulation([generate_base_population, model])

    mortality_rate = simulation.values.get_rate('mortality_rate')
    mortality_rate.source = simulation.tables.build_table(build_table(0.0))

    pump_simulation(simulation, iterations=1)

    # Folks instantly transition to sick so now our mortality rate should be much higher
    assert np.allclose(from_yearly(0.7, time_step), mortality_rate(simulation.population.population.index))


@patch('ceam_public_health.components.disease.get_disease_states')
def test_incidence(get_disease_states_mock):
    time_step = config.getfloat('simulation_parameters', 'time_step')
    time_step = timedelta(days=time_step)

    get_disease_states_mock.side_effect = lambda population, state_map: pd.DataFrame({'condition_state': 'healthy'}, index=population.index)
    model = DiseaseModel('test_disease')
    healthy = State('healthy')
    sick = State('sick')

    transition = RateTransition(sick, 'test_incidence', get_incidence(2412))
    healthy.transition_set.append(transition)

    model.states.extend([healthy, sick])

    simulation = setup_simulation([generate_base_population, model])

    transition.base_incidence = simulation.tables.build_table(build_table(0.7))

    incidence_rate = simulation.values.get_rate('incidence_rate.test_incidence')

    pump_simulation(simulation, iterations=1)

    assert np.all(from_yearly(0.7, time_step) == incidence_rate(simulation.population.population.index))


@patch('ceam_public_health.components.disease.get_disease_states')
def test_load_population_custom_columns(get_disease_states_mock):
    get_disease_states_mock.side_effect = lambda population, state_map: pd.DataFrame({'condition_state': 'healthy'}, index=population.index)
    model = DiseaseModel('test_disease')
    dwell_test = DiseaseState('dwell_test', disability_weight=0.0, dwell_time=10, event_time_column='special_test_time', event_count_column='special_test_count')

    model.states.append(dwell_test)

    simulation = setup_simulation([generate_base_population, model])

    assert 'special_test_time' in simulation.population.population
    assert 'special_test_count' in simulation.population.population
    assert np.all(simulation.population.population.special_test_count == 0)
    assert np.all(simulation.population.population.special_test_time == 0)


# End.
