import os
from datetime import timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd

from ceam import config
from ceam.framework.state_machine import Transition, State
from ceam.framework.util import from_yearly
from ceam.test_util import setup_simulation, pump_simulation, build_table, generate_test_population

from ceam_inputs import get_incidence


from ceam_public_health.disease import DiseaseState, ExcessMortalityState, RateTransition, DiseaseModel


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


@patch('ceam_public_health.disease.model.get_disease_states')
def test_dwell_time(get_disease_states_mock):
    config.simulation_parameters.set_with_metadata('time_step', 10, layer='override', source=os.path.realpath(__file__))
    get_disease_states_mock.side_effect = lambda population, state_map: pd.DataFrame(
        {'condition_state': 'healthy'}, index=population.index)
    model = DiseaseModel('state')
    healthy_state = State('healthy')
    event_state = DiseaseState('event', dwell_time=timedelta(days=28))
    healthy_state.transition_set.append(Transition(event_state))

    done_state = State('sick')
    event_state.transition_set.append(Transition(done_state))
    model.states.extend([healthy_state, event_state, done_state])

    simulation = setup_simulation([generate_test_population, model], population_size=10)

    # Move everyone into the event state
    event_time = simulation.current_time
    pump_simulation(simulation, iterations=1)

    assert np.all(simulation.population.population.state == 'event')

    # Not enough time has passed for people to move out of the event state, so they should all still be there
    pump_simulation(simulation, iterations=1)

    assert np.all(simulation.population.population.state == 'event')

    pump_simulation(simulation, iterations=1)

    # Now enough time has passed so people should transition away
    assert np.all(simulation.population.population.state == 'sick')
    assert np.all(simulation.population.population.event_event_time == pd.to_datetime(event_time))
    assert np.all(simulation.population.population.event_event_count == 1)


def test_mortality_rate():
    time_step = config.simulation_parameters.time_step
    time_step = timedelta(days=time_step)

    model = DiseaseModel('test_disease', csmr_data=build_table(0))
    healthy = State('healthy')
    mortality_state = ExcessMortalityState('sick',
                                           excess_mortality_data=build_table(0.7),
                                           disability_weight=0.1,
                                           prevalence_data=build_table(0.0000001, ['age', 'year', 'sex', 'prevalence']))

    healthy.transition_set.append(Transition(mortality_state))

    model.states.extend([healthy, mortality_state])

    simulation = setup_simulation([generate_test_population, model])

    mortality_rate = simulation.values.get_rate('mortality_rate')
    mortality_rate.source = simulation.tables.build_table(build_table(0.0))

    pump_simulation(simulation, iterations=1)

    # Folks instantly transition to sick so now our mortality rate should be much higher
    assert np.allclose(from_yearly(0.7, time_step), mortality_rate(simulation.population.population.index)['sick'])


@patch('ceam_public_health.disease.model.get_disease_states')
def test_incidence(get_disease_states_mock):
    time_step = config.simulation_parameters.time_step
    time_step = timedelta(days=time_step)

    get_disease_states_mock.side_effect = lambda population, state_map: pd.DataFrame(
        {'condition_state': 'healthy'}, index=population.index)
    model = DiseaseModel('test_disease')
    healthy = State('healthy')
    sick = State('sick')

    transition = RateTransition(sick, 'test_incidence', get_incidence(2412))
    healthy.transition_set.append(transition)

    model.states.extend([healthy, sick])

    simulation = setup_simulation([generate_test_population, model])

    transition.base_incidence = simulation.tables.build_table(build_table(0.7))

    incidence_rate = simulation.values.get_rate('incidence_rate.test_incidence')

    pump_simulation(simulation, iterations=1)

    assert np.allclose(from_yearly(0.7, time_step), incidence_rate(simulation.population.population.index), atol=0.00001)


@patch('ceam_public_health.disease.model.get_disease_states')
def test_risk_deletion(get_disease_states_mock):
    time_step = config.simulation_parameters.time_step
    time_step = timedelta(days=time_step)

    get_disease_states_mock.side_effect = lambda population, state_map: pd.DataFrame({'condition_state': 'healthy'},
                                                                                     index=population.index)
    model = DiseaseModel('test_disease')
    healthy = State('healthy')
    sick = State('sick')

    transition = RateTransition(sick, 'test_incidence', get_incidence(2412))
    healthy.transition_set.append(transition)

    model.states.extend([healthy, sick])

    simulation = setup_simulation([generate_test_population, model])

    base_rate = 0.7
    paf = 0.1
    transition.base_incidence = simulation.tables.build_table(build_table(base_rate))

    incidence_rate = simulation.values.get_rate('incidence_rate.test_incidence')

    simulation.values.mutator(simulation.tables.build_table(build_table(paf)), 'paf.{}'.format('test_incidence'))

    pump_simulation(simulation, iterations=1)

    expected_rate = base_rate * (1 - paf)

    assert np.allclose(from_yearly(expected_rate, time_step),
                       incidence_rate(simulation.population.population.index), atol=0.00001)


@patch('ceam_public_health.disease.model.get_disease_states')
def test_load_population_custom_columns(get_disease_states_mock):
    get_disease_states_mock.side_effect = lambda population, state_map: pd.DataFrame(
        {'condition_state': 'healthy'}, index=population.index)
    model = DiseaseModel('test_disease')
    dwell_test = DiseaseState('dwell_test', disability_weight=0.0, dwell_time=10,
                              event_time_column='special_test_time', event_count_column='special_test_count')

    model.states.append(dwell_test)

    simulation = setup_simulation([generate_test_population, model])

    assert 'special_test_time' in simulation.population.population
    assert 'special_test_count' in simulation.population.population
    assert np.all(simulation.population.population.special_test_count == 0)

    assert np.all(simulation.population.population.special_test_time.isnull())
