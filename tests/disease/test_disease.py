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
    base_config.simulation_parameters.set_with_metadata('year_start', 1990, **metadata)
    base_config.simulation_parameters.set_with_metadata('year_end', 2010, **metadata)
    base_config.simulation_parameters.set_with_metadata('time_step', 30.5, **metadata)
    return base_config


@pytest.fixture(scope='function')
def disease():
    Disease = namedtuple('Disease', 'name')
    return Disease(name='test')


@pytest.fixture(scope='function')
def assign_cause_mock(mocker):
    return mocker.patch('ceam_public_health.disease.model.assign_cause_at_beginning_of_simulation')


def test_dwell_time(assign_cause_mock, config, disease):
    time_step = 10
    assign_cause_mock.side_effect = lambda population, state_map: pd.DataFrame(
        {'condition_state': 'healthy'}, index=population.index)

    config.simulation_parameters.set_with_metadata('time_step', time_step,
                                                   layer='override', source=os.path.realpath(__file__))

    healthy_state = BaseDiseaseState('healthy')
    event_state = DiseaseState('event', get_data_functions={'dwell_time': lambda _, __: pd.Timedelta(days=28),
                                                            'disability_weight': lambda _, __: 0,
                                                            'prevalence': lambda _, __: None})
    done_state = BaseDiseaseState('sick')

    healthy_state.add_transition(event_state)
    event_state.add_transition(done_state)

    model = DiseaseModel(disease, states=[healthy_state, event_state, done_state],
                         get_data_functions={'csmr': lambda _, __: None})

    simulation = setup_simulation([TestPopulation(), model], population_size=10, input_config=config)

    # Move everyone into the event state

    pump_simulation(simulation, iterations=1)
    event_time = simulation.current_time
    assert np.all(simulation.population.population[disease.name] == 'event')

    pump_simulation(simulation, iterations=2)
    # Not enough time has passed for people to move out of the event state, so they should all still be there
    assert np.all(simulation.population.population[disease.name] == 'event')

    pump_simulation(simulation, iterations=1)
    # Now enough time has passed so people should transition away
    assert np.all(simulation.population.population[disease.name] == 'sick')
    assert np.all(simulation.population.population.event_event_time == pd.to_datetime(event_time))
    assert np.all(simulation.population.population.event_event_count == 1)


def test_mortality_rate(config, disease):
    year_start = config.simulation_parameters.year_start
    year_end = config.simulation_parameters.year_end

    time_step = pd.Timedelta(days=config.simulation_parameters.time_step)

    healthy = BaseDiseaseState('healthy')
    mort_get_data_funcs = {
        'dwell_time': lambda _, __: pd.Timedelta(days=0),
        'disability_weight': lambda _, __: 0.1,
        'prevalence': lambda _, __: build_table(0.000001, year_start, year_end,
                                                ['age', 'year', 'sex', 'prevalence']),
        'excess_mortality': lambda _, __: build_table(0.7, year_start, year_end),
    }
    mortality_state = ExcessMortalityState('sick', get_data_functions=mort_get_data_funcs)

    healthy.add_transition(mortality_state)

    model = DiseaseModel(disease, states=[healthy, mortality_state],
                         get_data_functions={'csmr': lambda _, __: None})

    simulation = setup_simulation([TestPopulation(), model], input_config=config)

    mortality_rate = simulation.values.register_rate_producer('mortality_rate')
    mortality_rate.source = simulation.tables.build_table(build_table(0.0, year_start, year_end))

    pump_simulation(simulation, iterations=1)

    # Folks instantly transition to sick so now our mortality rate should be much higher
    assert np.allclose(from_yearly(0.7, time_step), mortality_rate(simulation.population.population.index)['sick'])


def test_incidence(assign_cause_mock, config, disease):
    time_step = pd.Timedelta(days=config.simulation_parameters.time_step)

    assign_cause_mock.side_effect = lambda population, state_map: pd.DataFrame(
        {'condition_state': 'healthy'}, index=population.index)

    healthy = BaseDiseaseState('healthy')
    sick = BaseDiseaseState('sick')
    healthy.add_transition(sick)
    transition = RateTransition(
        input_state=healthy, output_state=sick,
        get_data_functions={
            'incidence': lambda _, __: get_incidence(sequelae.acute_myocardial_infarction_first_2_days, config)
        })
    healthy.transition_set.append(transition)

    model = DiseaseModel(disease, states=[healthy, sick],
                         get_data_functions={'csmr': lambda _, __: None})

    simulation = setup_simulation([TestPopulation(), model], input_config=config)
    year_start = config.simulation_parameters.year_start
    year_end = config.simulation_parameters.year_end
    transition.base_incidence = simulation.tables.build_table(build_table(0.7, year_start, year_end))

    incidence_rate = simulation.values.get_rate('sick.incidence_rate')

    pump_simulation(simulation, iterations=1)

    assert np.allclose(from_yearly(0.7, time_step),
                       incidence_rate(simulation.population.population.index), atol=0.00001)


def test_risk_deletion(assign_cause_mock, config, disease):
    time_step = config.simulation_parameters.time_step
    time_step = pd.Timedelta(days=time_step)
    year_start = config.simulation_parameters.year_start
    year_end = config.simulation_parameters.year_end

    assign_cause_mock.side_effect = lambda population, state_map: pd.DataFrame(
        {'condition_state': 'healthy'}, index=population.index)

    healthy = BaseDiseaseState('healthy')
    sick = BaseDiseaseState('sick')
    transition = RateTransition(
        input_state=healthy, output_state=sick,
        get_data_functions={
            'incidence': lambda _, __: get_incidence(sequelae.acute_myocardial_infarction_first_2_days, config)}
    )
    healthy.transition_set.append(transition)

    model = DiseaseModel(disease, states=[healthy, sick],
                         get_data_functions={'csmr': lambda _, __: None})

    simulation = setup_simulation([TestPopulation(), model], input_config=config)

    base_rate = 0.7
    paf = 0.1
    transition.base_incidence = simulation.tables.build_table(build_table(base_rate, year_start, year_end))

    incidence_rate = simulation.values.get_rate('sick.incidence_rate')

    simulation.values.register_value_modifier(
        'sick.paf', modifier=simulation.tables.build_table(build_table(paf, year_start, year_end)))

    pump_simulation(simulation, iterations=1)

    expected_rate = base_rate * (1 - paf)

    assert np.allclose(from_yearly(expected_rate, time_step),
                       incidence_rate(simulation.population.population.index), atol=0.00001)
