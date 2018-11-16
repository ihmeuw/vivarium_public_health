import numpy as np
import pandas as pd
import pytest

from vivarium.framework.utilities import from_yearly
from vivarium.testing_utilities import build_table, TestPopulation, metadata
from vivarium.interface.interactive import setup_simulation

from vivarium_public_health.disease import (BaseDiseaseState, DiseaseState, ExcessMortalityState,
                                            RateTransition, DiseaseModel)
from vivarium_public_health.population import Mortality


@pytest.fixture
def disease():
    return 'test'


@pytest.fixture
def assign_cause_mock(mocker):
    return mocker.patch('vivarium_public_health.disease.model.DiseaseModel.assign_initial_status_to_simulants')


@pytest.fixture
def base_data():
    def _set_prevalence(p):
        base_function = dict()
        base_function['disability_weight'] = lambda _, __: 0
        base_function['dwell_time'] = lambda _, __: pd.Timedelta(days=1)
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


def test_dwell_time(assign_cause_mock, base_config, disease, base_data):
    time_step = 10
    assign_cause_mock.side_effect = lambda population, *args: pd.DataFrame(
        {'condition_state': 'healthy'}, index=population.index)

    base_config.update({
        'time': {'step_size': time_step},
        'population': {'population_size': 10}
    }, **metadata(__file__))

    healthy_state = BaseDiseaseState('healthy')
    data_function = base_data(0)
    data_function['dwell_time'] = lambda _, __: pd.Timedelta(days=28)
    event_state = DiseaseState('event', get_data_functions=data_function)
    done_state = BaseDiseaseState('sick')

    healthy_state.add_transition(event_state)
    event_state.add_transition(done_state)

    model = DiseaseModel(disease, initial_state=healthy_state, states=[healthy_state, event_state, done_state],
                         get_data_functions={'csmr': lambda _, __: None})

    simulation = setup_simulation([TestPopulation(), model], base_config)

    # Move everyone into the event state
    simulation.step()
    event_time = simulation.clock.time
    assert np.all(simulation.population.population[disease] == 'event')

    simulation.step()
    simulation.step()
    # Not enough time has passed for people to move out of the event state, so they should all still be there
    assert np.all(simulation.population.population[disease] == 'event')

    simulation.step()
    # Now enough time has passed so people should transition away
    assert np.all(simulation.population.population[disease] == 'sick')
    assert np.all(simulation.population.population.event_event_time == pd.to_datetime(event_time))
    assert np.all(simulation.population.population.event_event_count == 1)


def test_dwell_time_with_mortality(base_config, base_plugins, disease):
    year_start = base_config.time.start.year
    year_end = base_config.time.end.year

    time_step = 10
    pop_size = 100
    base_config.update({
        'time': {'step_size': time_step},
        'population': {'population_size': pop_size}
    }, **metadata(__file__))
    healthy_state = BaseDiseaseState('healthy')

    mort_get_data_funcs = {
        'dwell_time': lambda _, __: pd.Timedelta(days=14),
        'excess_mortality': lambda _, __: build_table(0.7, year_start-1, year_end),
    }

    mortality_state = ExcessMortalityState('event', get_data_functions=mort_get_data_funcs)
    done_state = BaseDiseaseState('sick')

    healthy_state.add_transition(mortality_state)
    mortality_state.add_transition(done_state)

    model = DiseaseModel(disease, initial_state=healthy_state, states=[healthy_state, mortality_state, done_state],
                         get_data_functions={'csmr': lambda _, __: None})
    mortality = Mortality()
    simulation = setup_simulation([TestPopulation(), model, mortality], base_config, base_plugins)

    # Move everyone into the event state
    simulation.step()
    assert np.all(simulation.population.population[disease] == 'event')

    simulation.step()
    # Not enough time has passed for people to move out of the event state, so they should all still be there
    assert np.all(simulation.population.population[disease] == 'event')

    simulation.step()

    # Make sure some people have died and remained in event state
    assert (simulation.population.population['alive'] == 'alive').sum() < pop_size

    assert ((simulation.population.population['alive'] == 'dead').sum() ==
            (simulation.population.population[disease] == 'event').sum())

    # enough time has passed so living people should transition away to sick
    assert ((simulation.population.population['alive'] == 'alive').sum() ==
           (simulation.population.population[disease] == 'sick').sum())



@pytest.mark.parametrize('test_prevalence_level', [0, 0.35, 1])
def test_prevalence_single_state_with_migration(base_config, disease, base_data, test_prevalence_level):
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
    base_config.update({'population': {'population_size': 50000}}, **metadata(__file__))
    simulation = setup_simulation([TestPopulation(), model], base_config)
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
def test_prevalence_multiple_sequelae(base_config, disease, base_data, test_prevalence_level):
    healthy = BaseDiseaseState('healthy')

    sequela = dict()
    for i, p in enumerate(test_prevalence_level):
        sequela[i] = DiseaseState('sequela'+str(i), get_data_functions=base_data(p))

    model = DiseaseModel(disease, initial_state=healthy, states=[healthy, sequela[0], sequela[1], sequela[2]],
                         get_data_functions={'csmr': lambda _, __: None})
    base_config.update({'population': {'population_size': 100000}}, **metadata(__file__))
    simulation = setup_simulation([TestPopulation(), model], base_config)
    error_message = "initial sequela status of simulants should be matched to the prevalence data."
    assert np.allclose([get_test_prevalence(simulation, 'sequela0'),
                        get_test_prevalence(simulation, 'sequela1'),
                        get_test_prevalence(simulation, 'sequela2')], test_prevalence_level, .02), error_message


def test_prevalence_single_simulant():
    # pandas has a bug on the case of single element with non-zero index; this test is to catch that case
    test_index = [20]
    initial_state = 'healthy'
    simulants_df = pd.DataFrame({'sex': 'Female', 'age': 3, 'sex_id': 2.0}, index=test_index)
    state_names = ['sick', 'healthy']
    weights = np.array([[1, 1]])
    simulants = DiseaseModel.assign_initial_status_to_simulants(simulants_df, state_names, weights,
                                                                pd.Series(0.5, index=test_index))
    expected = simulants_df[['age', 'sex']]
    expected['condition_state'] = 'sick'
    assert expected.equals(simulants)


def test_mortality_rate(base_config, base_plugins, disease):
    year_start = base_config.time.start.year
    year_end = base_config.time.end.year

    time_step = pd.Timedelta(days=base_config.time.step_size)

    healthy = BaseDiseaseState('healthy')
    mort_get_data_funcs = {
        'dwell_time': lambda _, __: pd.Timedelta(days=0),
        'disability_weight': lambda _, __: 0.1,
        'prevalence': lambda _, __: build_table(0.000001, year_start-1, year_end,
                                                ['age', 'year', 'sex', 'value']),
        'excess_mortality': lambda _, __: build_table(0.7, year_start-1, year_end),
    }

    mortality_state = ExcessMortalityState('sick', get_data_functions=mort_get_data_funcs)

    healthy.add_transition(mortality_state)

    model = DiseaseModel(disease, initial_state=healthy, states=[healthy, mortality_state],
                         get_data_functions={'csmr': lambda _, __: None})

    simulation = setup_simulation([TestPopulation(), model], base_config, base_plugins)

    mortality_rate = simulation.values.register_rate_producer('mortality_rate')
    mortality_rate.source = simulation.tables.build_table(build_table(0.0, year_start, year_end),
                                                          key_columns=('sex',),
                                                          parameter_columns=[('age', 'age_group_start', 'age_group_end'),
                                                                             ('year', 'year_start', 'year_end')],
                                                          value_columns=None)

    simulation.step()
    # Folks instantly transition to sick so now our mortality rate should be much higher
    assert np.allclose(from_yearly(0.7, time_step), mortality_rate(simulation.population.population.index)['sick'])


def test_incidence(base_config, base_plugins, disease):
    year_start = base_config.time.start.year
    year_end = base_config.time.end.year
    time_step = pd.Timedelta(days=base_config.time.step_size)

    healthy = BaseDiseaseState('healthy')
    sick = BaseDiseaseState('sick')
    healthy.add_transition(sick)

    transition = RateTransition(
        input_state=healthy, output_state=sick,
        get_data_functions={
            'incidence_rate': lambda _, builder: builder.data.load(
                f"sequela.acute_myocardial_infarction_first_2_days.incidence")
        })
    healthy.transition_set.append(transition)

    model = DiseaseModel(disease, initial_state=healthy, states=[healthy, sick],
                         get_data_functions={'csmr': lambda _, __: None})

    simulation = setup_simulation([TestPopulation(), model], base_config, base_plugins)

    transition.base_rate = simulation.tables.build_table(build_table(0.7, year_start, year_end),
                                                         key_columns=('sex',),
                                                         parameter_columns=[('age', 'age_group_start', 'age_group_end'),
                                                                            ('year', 'year_start', 'year_end')],
                                                         value_columns=None)

    incidence_rate = simulation.values.get_rate('sick.incidence_rate')

    simulation.step()

    assert np.allclose(from_yearly(0.7, time_step),
                       incidence_rate(simulation.population.population.index), atol=0.00001)


def test_risk_deletion(base_config, base_plugins, disease):
    time_step = base_config.time.step_size
    time_step = pd.Timedelta(days=time_step)
    year_start = base_config.time.start.year
    year_end = base_config.time.end.year

    healthy = BaseDiseaseState('healthy')
    sick = BaseDiseaseState('sick')
    transition = RateTransition(
        input_state=healthy, output_state=sick,
        get_data_functions={
            'incidence_rate': lambda _, builder: builder.data.load(
                f"sequela.acute_myocardial_infarction_first_2_days.incidence")
        }
    )
    healthy.transition_set.append(transition)

    model = DiseaseModel(disease, initial_state=healthy, states=[healthy, sick],
                         get_data_functions={'csmr': lambda _, __: None})

    simulation = setup_simulation([TestPopulation(), model], base_config, base_plugins)

    base_rate = 0.7
    paf = 0.1
    transition.base_rate = simulation.tables.build_table(build_table(base_rate, year_start, year_end),
                                                         key_columns=('sex',),
                                                         parameter_columns=[('age', 'age_group_start', 'age_group_end'),
                                                                            ('year', 'year_start', 'year_end')],
                                                         value_columns=None)

    incidence_rate = simulation.values.get_rate('sick.incidence_rate')

    simulation.values.register_value_modifier(
        'sick.paf', modifier=simulation.tables.build_table(build_table(paf, year_start, year_end),
                                                           key_columns=('sex',),
                                                           parameter_columns=[('age', 'age_group_start', 'age_group_end'),
                                                                              ('year', 'year_start', 'year_end')],
                                                           value_columns=None))

    simulation.step()

    expected_rate = base_rate * (1 - paf)

    assert np.allclose(from_yearly(expected_rate, time_step),
                       incidence_rate(simulation.population.population.index), atol=0.00001)


def test__assign_event_time_for_prevalent_cases():
    pop_data = pd.DataFrame(index=range(100))
    random_func = lambda index: pd.Series(0.4, index=index)
    current_time = pd.Timestamp(2017, 1, 10, 12)

    dwell_time_func = lambda index: pd.Series(10, index=index)
    # 10* 0.4 = 4 ; 4 days before the current time
    expected = pd.Series(pd.Timestamp(2017, 1, 6, 12), index=pop_data.index)

    assert expected.equals(DiseaseState._assign_event_time_for_prevalent_cases(pop_data, current_time, random_func,
                                                                               dwell_time_func))