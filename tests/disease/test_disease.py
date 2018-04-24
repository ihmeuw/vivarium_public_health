import os
from collections import namedtuple

import numpy as np
import pandas as pd
import pytest

from vivarium.framework.util import from_yearly

from vivarium.test_util import setup_simulation, pump_simulation, build_table, TestPopulation

from ceam_inputs import get_incidence, sequelae

from ceam_public_health.disease import (BaseDiseaseState, DiseaseState, ExcessMortalityState,
                                        RateTransition, DiseaseModel)


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
    base_config.randomness.update({'key_columns': ['entrance_time', 'age']}, **metadata)
    return base_config


@pytest.fixture(scope='function')
def disease():
    Disease = namedtuple('Disease', 'name')
    return Disease(name='test')


@pytest.fixture(scope='function')
def assign_cause_mock(mocker):
    return mocker.patch('ceam_public_health.disease.model.DiseaseModel.assign_initial_status_to_simulants')


@pytest.fixture(scope='function')
def base_data():
    def _set_prevalence(p):
        base_function = dict()
        base_function['disability_weight'] = lambda _, __: 0
        base_function['dwell_time'] = lambda _, __: pd.Timedelta(days=0)
        base_function['prevalence'] = lambda _, __: p
        return base_function
    return _set_prevalence


def get_test_prevalence(simulation, key):
    """
    Helper function to calculate the prevalence for the given state(key)
    """
    try:
        simulants_status_counts = simulation.population.population.test.value_counts().to_dict()
        result = float(simulants_status_counts[key] / simulation.population.population.test.size)
    except KeyError:
        result = 0
    return result


def test_dwell_time(assign_cause_mock, config, disease, base_data):
    time_step = 10
    assign_cause_mock.side_effect = lambda population, *args: pd.DataFrame(
        {'condition_state': 'healthy'}, index=population.index)

    config.time.set_with_metadata('step_size', time_step, layer='override', source=os.path.realpath(__file__))

    healthy_state = BaseDiseaseState('healthy')
    data_function = base_data(0)
    data_function['dwell_time'] = lambda _, __: pd.Timedelta(days=28)
    event_state = DiseaseState('event', get_data_functions=data_function)
    done_state = BaseDiseaseState('sick')

    healthy_state.add_transition(event_state)
    event_state.add_transition(done_state)

    model = DiseaseModel(disease, initial_state=healthy_state, states=[healthy_state, event_state, done_state],
                         get_data_functions={'csmr': lambda _, __: None})

    simulation = setup_simulation([TestPopulation(), model], population_size=10, input_config=config)

    # Move everyone into the event state
    pump_simulation(simulation, iterations=1)
    event_time = simulation.clock.time
    assert np.all(simulation.population.population[disease.name] == 'event')
    pump_simulation(simulation, iterations=2)
    # Not enough time has passed for people to move out of the event state, so they should all still be there
    assert np.all(simulation.population.population[disease.name] == 'event')
    pump_simulation(simulation, iterations=1)
    # Now enough time has passed so people should transition away
    assert np.all(simulation.population.population[disease.name] == 'sick')
    assert np.all(simulation.population.population.event_event_time == pd.to_datetime(event_time))
    assert np.all(simulation.population.population.event_event_count == 1)


@pytest.mark.parametrize('test_prevalence_level', [0, 0.35, 1])
def test_prevalence_single_state_with_migration(config, disease, base_data, test_prevalence_level):
    """
    Test the prevalence for the single state over newly migrated population.
    Start with the initial population, check the prevalence for initial assignment.
    Then add new simulants and check whether the initial status is
    properly assigned to new simulants based on the prevalence data and pre-existing simulants status

    """
    healthy = BaseDiseaseState('healthy')

    sick = DiseaseState('sick', get_data_functions=base_data(test_prevalence_level))
    model = DiseaseModel(disease, initial_state=healthy, states=[healthy, sick],
                         get_data_functions={'csmr': lambda _, __: None})
    simulation = setup_simulation([TestPopulation(), model], population_size=50000, input_config=config)
    error_message = "initial status of simulants should be matched to the prevalence data."
    assert np.isclose(get_test_prevalence(simulation, 'sick'), test_prevalence_level, 0.01), error_message
    simulation.clock.step_forward()
    assert np.isclose(get_test_prevalence(simulation, 'sick'), test_prevalence_level, .01), error_message
    simulation.simulant_creator(50000)
    assert np.isclose(get_test_prevalence(simulation, 'sick'), test_prevalence_level, .01), error_message
    simulation.clock.step_forward()
    simulation.simulant_creator(50000)
    assert np.isclose(get_test_prevalence(simulation, 'sick'), test_prevalence_level, .01), error_message


@pytest.mark.parametrize('test_prevalence_level',
                         [[0.15, 0.05, 0.35], [0, 0.15, 0.5], [0.2, 0.3, 0.5], [0, 0, 1], [0, 0, 0]])
def test_prevalence_multiple_sequelae(config, disease, base_data, test_prevalence_level):
    healthy = BaseDiseaseState('healthy')

    sequela = dict()
    for i, p in enumerate(test_prevalence_level):
        sequela[i] = DiseaseState('sequela'+str(i), get_data_functions=base_data(p))

    model = DiseaseModel(disease, initial_state=healthy, states=[healthy, sequela[0], sequela[1], sequela[2]],
                         get_data_functions={'csmr': lambda _, __: None})
    simulation = setup_simulation([TestPopulation(), model], population_size=100000, input_config=config)
    error_message = "initial sequela status of simulants should be matched to the prevalence data."
    assert np.allclose([get_test_prevalence(simulation, 'sequela0'),
                        get_test_prevalence(simulation, 'sequela1'),
                        get_test_prevalence(simulation, 'sequela2')],test_prevalence_level, .02), error_message


def test_mortality_rate(config, disease):
    year_start = config.time.start.year
    year_end = config.time.end.year

    time_step = pd.Timedelta(days=config.time.step_size)

    healthy = BaseDiseaseState('healthy')
    mort_get_data_funcs = {
        'dwell_time': lambda _, __: pd.Timedelta(days=0),
        'disability_weight': lambda _, __: 0.1,
        'prevalence': lambda _, __: 1,
        'excess_mortality': lambda _, __: 0.7,
    }

    mortality_state = ExcessMortalityState('sick', get_data_functions=mort_get_data_funcs)

    healthy.add_transition(mortality_state)

    model = DiseaseModel(disease, initial_state=healthy, states=[healthy, mortality_state],
                         get_data_functions={'csmr': lambda _, __: None})

    simulation = setup_simulation([TestPopulation(), model], input_config=config)

    mortality_rate = simulation.values.register_rate_producer('mortality_rate')
    mortality_rate.source = simulation.tables.build_table(build_table(0.0, year_start, year_end))

    pump_simulation(simulation, iterations=1)
    # Folks instantly transition to sick so now our mortality rate should be much higher
    assert np.allclose(from_yearly(0.7, time_step), mortality_rate(simulation.population.population.index)['sick'])


def test_incidence(config, disease):
    time_step = pd.Timedelta(days=config.time.step_size)

    healthy = BaseDiseaseState('healthy')
    sick = BaseDiseaseState('sick')
    healthy.add_transition(sick)

    transition = RateTransition(
        input_state=healthy, output_state=sick,
        get_data_functions={
            'incidence_rate': lambda _, __: get_incidence(sequelae.acute_myocardial_infarction_first_2_days, config)
        })
    healthy.transition_set.append(transition)

    model = DiseaseModel(disease, initial_state=healthy, states=[healthy, sick],
                         get_data_functions={'csmr': lambda _, __: None})

    simulation = setup_simulation([TestPopulation(), model], input_config=config)
    year_start = config.time.start.year
    year_end = config.time.end.year
    transition.base_rate = simulation.tables.build_table(build_table(0.7, year_start, year_end))

    incidence_rate = simulation.values.get_rate('sick.incidence_rate')

    pump_simulation(simulation, iterations=1)

    assert np.allclose(from_yearly(0.7, time_step),
                       incidence_rate(simulation.population.population.index), atol=0.00001)


def test_risk_deletion(config, disease):
    time_step = config.time.step_size
    time_step = pd.Timedelta(days=time_step)
    year_start = config.time.start.year
    year_end = config.time.end.year

    healthy = BaseDiseaseState('healthy')
    sick = BaseDiseaseState('sick')
    transition = RateTransition(
        input_state=healthy, output_state=sick,
        get_data_functions={
            'incidence_rate': lambda _, __: get_incidence(sequelae.acute_myocardial_infarction_first_2_days, config)}
    )
    healthy.transition_set.append(transition)

    model = DiseaseModel(disease, initial_state=healthy, states=[healthy, sick],
                         get_data_functions={'csmr': lambda _, __: None})

    simulation = setup_simulation([TestPopulation(), model], input_config=config)

    base_rate = 0.7
    paf = 0.1
    transition.base_rate = simulation.tables.build_table(build_table(base_rate, year_start, year_end))

    incidence_rate = simulation.values.get_rate('sick.incidence_rate')

    simulation.values.register_value_modifier(
        'sick.paf', modifier=simulation.tables.build_table(build_table(paf, year_start, year_end)))

    pump_simulation(simulation, iterations=1)

    expected_rate = base_rate * (1 - paf)

    assert np.allclose(from_yearly(expected_rate, time_step),
                       incidence_rate(simulation.population.population.index), atol=0.00001)
