import numpy as np
import pandas as pd
import pytest
from vivarium import Component, InteractiveContext
from vivarium.framework.state_machine import Transition
from vivarium.framework.utilities import from_yearly
from vivarium.testing_utilities import TestPopulation, metadata

from tests.test_utilities import build_table_with_age
from vivarium_public_health.disease import (
    BaseDiseaseState,
    DiseaseModel,
    DiseaseState,
    RateTransition,
)
from vivarium_public_health.disease.state import SusceptibleState
from vivarium_public_health.disease.transition import TransitionString
from vivarium_public_health.population import Mortality


@pytest.fixture
def disease():
    return "test"


@pytest.fixture
def assign_cause_mock(mocker):
    return mocker.patch(
        "vivarium_public_health.disease.model.DiseaseModel.assign_initial_status_to_simulants"
    )


@pytest.fixture
def base_data():
    def _set_prevalence(p):
        base_function = dict()
        base_function["dwell_time"] = lambda _, __: pd.Timedelta(days=1)
        base_function["prevalence"] = lambda _, __: p
        return base_function

    return _set_prevalence


def get_test_prevalence(simulation, key):
    """
    Helper function to calculate the prevalence for the given state(key)
    """
    try:
        simulants_status_counts = simulation.get_population().test.value_counts().to_dict()
        result = float(simulants_status_counts[key] / simulation.get_population().test.size)
    except KeyError:
        result = 0
    return result


def test_dwell_time(assign_cause_mock, base_config, base_plugins, disease, base_data):
    time_step = 10
    assign_cause_mock.side_effect = lambda population, *args: pd.DataFrame(
        {"condition_state": "healthy"}, index=population.index
    )

    base_config.update(
        {"time": {"step_size": time_step}, "population": {"population_size": 10}},
        **metadata(__file__),
    )
    healthy_state = BaseDiseaseState("healthy")
    data_function = base_data(0)
    data_function["dwell_time"] = lambda _, __: pd.Timedelta(days=28)
    data_function["disability_weight"] = lambda _, __: 0.0
    event_state = DiseaseState("event", get_data_functions=data_function)
    done_state = BaseDiseaseState("sick")

    healthy_state.add_transition(Transition(healthy_state, event_state))
    event_state.add_dwell_time_transition(done_state)

    model = DiseaseModel(
        disease, initial_state=healthy_state, states=[healthy_state, event_state, done_state]
    )

    simulation = InteractiveContext(
        components=[TestPopulation(), model],
        configuration=base_config,
        plugin_configuration=base_plugins,
    )

    # Move everyone into the event state
    simulation.step()
    event_time = simulation._clock.time
    assert np.all(simulation.get_population()[disease] == "event")

    simulation.step()
    simulation.step()
    # Not enough time has passed for people to move out of the event state, so they should all still be there
    assert np.all(simulation.get_population()[disease] == "event")

    simulation.step()
    # Now enough time has passed so people should transition away
    assert np.all(simulation.get_population()[disease] == "sick")
    assert np.all(simulation.get_population().event_event_time == pd.to_datetime(event_time))
    assert np.all(simulation.get_population().event_event_count == 1)


def test_dwell_time_with_mortality(base_config, base_plugins, disease):
    year_start = base_config.time.start.year
    year_end = base_config.time.end.year

    time_step = 10
    pop_size = 100
    base_config.update(
        {"time": {"step_size": time_step}, "population": {"population_size": pop_size}},
        **metadata(__file__),
    )
    healthy_state = BaseDiseaseState("healthy")

    mort_get_data_funcs = {
        "dwell_time": lambda _, __: pd.Timedelta(days=14),
        "excess_mortality_rate": lambda _, __: build_table_with_age(
            0.7,
            parameter_columns={"year": (year_start - 1, year_end)},
        ),
        "disability_weight": lambda _, __: 0.0,
    }

    mortality_state = DiseaseState("event", get_data_functions=mort_get_data_funcs)
    done_state = BaseDiseaseState("sick")

    healthy_state.add_transition(Transition(healthy_state, mortality_state))
    mortality_state.add_dwell_time_transition(done_state)

    model = DiseaseModel(
        disease,
        initial_state=healthy_state,
        states=[healthy_state, mortality_state, done_state],
    )
    mortality = Mortality()
    simulation = InteractiveContext(
        components=[TestPopulation(), model, mortality],
        configuration=base_config,
        plugin_configuration=base_plugins,
    )

    # Move everyone into the event state
    simulation.step()
    assert np.all(simulation.get_population()[disease] == "event")

    simulation.step()
    # Not enough time has passed for people to move out of the event state, so they should all still be there
    assert np.all(simulation.get_population()[disease] == "event")

    simulation.step()

    # Make sure some people have died and remained in event state
    assert (simulation.get_population()["alive"] == "alive").sum() < pop_size

    assert (simulation.get_population()["alive"] == "dead").sum() == (
        simulation.get_population()[disease] == "event"
    ).sum()

    # enough time has passed so living people should transition away to sick
    assert (simulation.get_population()["alive"] == "alive").sum() == (
        simulation.get_population()[disease] == "sick"
    ).sum()


@pytest.mark.parametrize("test_prevalence_level", [0, 0.35, 1])
def test_prevalence_single_state_with_migration(
    base_config, base_plugins, disease, base_data, test_prevalence_level
):
    """
    Test the prevalence for the single state over newly migrated population.
    Start with the initial population, check the prevalence for initial assignment.
    Then add new simulants and check whether the initial status is
    properly assigned to new simulants based on the prevalence data and pre-existing simulants status

    """
    year_start = base_config.time.start.year
    year_end = base_config.time.end.year

    healthy = BaseDiseaseState("healthy")
    data_funcs = base_data(test_prevalence_level)
    data_funcs.update({"disability_weight": lambda _, __: 0.0})
    sick = DiseaseState("sick", get_data_functions=data_funcs)
    model = DiseaseModel(disease, initial_state=healthy, states=[healthy, sick])
    base_config.update({"population": {"population_size": 50000}}, **metadata(__file__))
    simulation = InteractiveContext(
        components=[TestPopulation(), model],
        configuration=base_config,
        plugin_configuration=base_plugins,
    )
    error_message = "initial status of simulants should be matched to the prevalence data."
    assert np.isclose(
        get_test_prevalence(simulation, "sick"), test_prevalence_level, 0.01
    ), error_message
    simulation.step()
    assert np.isclose(
        get_test_prevalence(simulation, "sick"), test_prevalence_level, 0.01
    ), error_message
    simulation.simulant_creator(
        50000,
        population_configuration={"age_start": 0, "age_end": 5, "sim_state": "time_step"},
    )
    assert np.isclose(
        get_test_prevalence(simulation, "sick"), test_prevalence_level, 0.01
    ), error_message
    simulation.step()
    simulation.simulant_creator(
        50000,
        population_configuration={"age_start": 0, "age_end": 5, "sim_state": "time_step"},
    )
    assert np.isclose(
        get_test_prevalence(simulation, "sick"), test_prevalence_level, 0.01
    ), error_message


@pytest.mark.parametrize(
    "test_prevalence_level",
    [[0.15, 0.05, 0.35], [0, 0.15, 0.5], [0.2, 0.3, 0.5], [0, 0, 1], [0, 0, 0]],
)
def test_prevalence_multiple_sequelae(
    base_config, base_plugins, disease, base_data, test_prevalence_level
):
    year_start = base_config.time.start.year
    year_end = base_config.time.end.year

    healthy = BaseDiseaseState("healthy")

    sequela = dict()
    for i, p in enumerate(test_prevalence_level):
        data_funcs = base_data(p)
        data_funcs.update({"disability_weight": lambda _, __: 0.0})
        sequela[i] = DiseaseState("sequela" + str(i), get_data_functions=data_funcs)

    model = DiseaseModel(
        disease, initial_state=healthy, states=[healthy, sequela[0], sequela[1], sequela[2]]
    )
    base_config.update({"population": {"population_size": 100000}}, **metadata(__file__))
    simulation = InteractiveContext(
        components=[TestPopulation(), model],
        configuration=base_config,
        plugin_configuration=base_plugins,
    )
    error_message = (
        "initial sequela status of simulants should be matched to the prevalence data."
    )
    assert np.allclose(
        [
            get_test_prevalence(simulation, "sequela0"),
            get_test_prevalence(simulation, "sequela1"),
            get_test_prevalence(simulation, "sequela2"),
        ],
        test_prevalence_level,
        0.02,
    ), error_message


def test_prevalence_single_simulant():
    # pandas has a bug on the case of single element with non-zero index; this test is to catch that case
    test_index = [20]
    initial_state = "healthy"
    simulants_df = pd.DataFrame({"sex": "Female", "age": 3, "sex_id": 2.0}, index=test_index)
    state_names = ["sick", "healthy"]
    weights = np.array([[1, 1]])
    simulants = DiseaseModel.assign_initial_status_to_simulants(
        simulants_df, state_names, weights, pd.Series(0.5, index=test_index)
    )
    expected = simulants_df[["age", "sex"]]
    expected["condition_state"] = "sick"
    assert expected.equals(simulants)


def test_mortality_rate(base_config, base_plugins, disease):
    year_start = base_config.time.start.year
    year_end = base_config.time.end.year

    time_step = pd.Timedelta(days=base_config.time.step_size)

    healthy = BaseDiseaseState("healthy")
    mort_get_data_funcs = {
        "dwell_time": lambda _, __: pd.Timedelta(days=0),
        "disability_weight": lambda _, __: 0.0,
        "prevalence": lambda _, __: build_table_with_age(
            0.000001,
            parameter_columns={"year": (year_start - 1, year_end)},
        ),
        "excess_mortality_rate": lambda _, __: build_table_with_age(
            0.7,
            parameter_columns={"year": (year_start - 1, year_end)},
        ),
    }

    mortality_state = DiseaseState("sick", get_data_functions=mort_get_data_funcs)

    healthy.add_transition(Transition(healthy, mortality_state))

    model = DiseaseModel(disease, initial_state=healthy, states=[healthy, mortality_state])

    simulation = InteractiveContext(
        components=[TestPopulation(), model, Mortality()],
        configuration=base_config,
        plugin_configuration=base_plugins,
    )

    mortality_rate = simulation._values.get_value("mortality_rate")

    simulation.step()
    # Folks instantly transition to sick so now our mortality rate should be much higher
    assert np.allclose(
        from_yearly(0.7, time_step), mortality_rate(simulation.get_population().index)["sick"]
    )


def test_incidence(base_config, base_plugins, disease):
    time_step = pd.Timedelta(days=base_config.time.step_size)

    healthy = BaseDiseaseState("healthy")
    sick = BaseDiseaseState("sick")

    key = f"sequela.acute_myocardial_infarction_first_2_days.incidence_rate"
    transition = RateTransition(
        input_state=healthy,
        output_state=sick,
        get_data_functions={"incidence_rate": lambda builder, _: builder.data.load(key)},
    )
    healthy.transition_set.append(transition)

    model = DiseaseModel(disease, initial_state=healthy, states=[healthy, sick])

    simulation = InteractiveContext(
        components=[TestPopulation(), model],
        configuration=base_config,
        plugin_configuration=base_plugins,
        setup=False,
    )
    simulation._data.write(key, 0.7)
    simulation.setup()

    incidence_rate = simulation._values.get_value("sick.incidence_rate")

    simulation.step()

    assert np.allclose(
        from_yearly(0.7, time_step),
        incidence_rate(simulation.get_population().index),
        atol=0.00001,
    )


def test_risk_deletion(base_config, base_plugins, disease):
    time_step = base_config.time.step_size
    time_step = pd.Timedelta(days=time_step)
    year_start = base_config.time.start.year
    year_end = base_config.time.end.year
    base_rate = 0.7
    paf = 0.1

    healthy = BaseDiseaseState("healthy")
    sick = BaseDiseaseState("sick")
    key = "sequela.acute_myocardial_infarction_first_2_days.incidence_rate"
    transition = RateTransition(
        input_state=healthy,
        output_state=sick,
        get_data_functions={"incidence_rate": lambda builder, _: builder.data.load(key)},
    )
    healthy.transition_set.append(transition)

    model = DiseaseModel(disease, initial_state=healthy, states=[healthy, sick])

    class PafModifier(Component):
        def setup(self, builder):
            builder.value.register_value_modifier(
                "sick.incidence_rate.paf",
                modifier=simulation._tables.build_table(
                    build_table_with_age(
                        paf,
                        parameter_columns={"year": (year_start, year_end)},
                    ),
                    key_columns=("sex",),
                    parameter_columns=["age", "year"],
                    value_columns=(),
                ),
            )

    simulation = InteractiveContext(
        components=[TestPopulation(), model, PafModifier()],
        configuration=base_config,
        plugin_configuration=base_plugins,
        setup=False,
    )
    simulation._data.write(key, base_rate)
    simulation.setup()

    incidence_rate = simulation._values.get_value("sick.incidence_rate")

    simulation.step()

    expected_rate = base_rate * (1 - paf)
    assert np.allclose(
        from_yearly(expected_rate, time_step),
        incidence_rate(simulation.get_population().index),
        atol=0.00001,
    )


def test__assign_event_time_for_prevalent_cases():
    pop_data = pd.DataFrame(index=range(100))
    random_func = lambda index: pd.Series(0.4, index=index)
    current_time = pd.Timestamp(2017, 1, 10, 12)

    dwell_time_func = lambda index: pd.Series(10, index=index)
    # 10* 0.4 = 4 ; 4 days before the current time
    expected = pd.Series(pd.Timestamp(2017, 1, 6, 12), index=pop_data.index)
    actual = DiseaseState._assign_event_time_for_prevalent_cases(
        pop_data, current_time, random_func, dwell_time_func
    )
    assert (expected == actual).all()


def test_prevalence_birth_prevalence_initial_assignment(base_config, base_plugins, disease):
    healthy = BaseDiseaseState("healthy")

    data_funcs = {
        "prevalence": lambda _, __: 1,
        "birth_prevalence": lambda _, __: 0.5,
        "disability_weight": lambda _, __: 0,
    }
    with_condition = DiseaseState("with_condition", get_data_functions=data_funcs)

    model = DiseaseModel(disease, initial_state=healthy, states=[healthy, with_condition])

    pop_size = 2000
    base_config.update(
        {"population": {"population_size": pop_size, "age_start": 0, "age_end": 5}},
        **metadata(__file__),
    )
    simulation = InteractiveContext(
        components=[TestPopulation(), model],
        configuration=base_config,
        plugin_configuration=base_plugins,
    )

    # prevalence should be used for assigning initial status at sim start
    assert np.isclose(get_test_prevalence(simulation, "with_condition"), 1)

    # birth prevalence should be used for assigning initial status to newly-borns on time steps
    simulation.step()
    simulation.simulant_creator(
        pop_size,
        population_configuration={"age_start": 0, "age_end": 0, "sim_state": "time_step"},
    )
    assert np.isclose(get_test_prevalence(simulation, "with_condition"), 0.75, 0.01)

    # and prevalence should be used for ages not start = end = 0
    simulation.step()
    simulation.simulant_creator(
        pop_size,
        population_configuration={"age_start": 0, "age_end": 5, "sim_state": "time_step"},
    )
    assert np.isclose(get_test_prevalence(simulation, "with_condition"), 0.83, 0.01)


def test_no_birth_prevalence_initial_assignment(base_config, base_plugins, disease):
    healthy = BaseDiseaseState("healthy")

    data_funcs = {"prevalence": lambda _, __: 1, "disability_weight": lambda _, __: 0}
    with_condition = DiseaseState("with_condition", get_data_functions=data_funcs)

    model = DiseaseModel(disease, initial_state=healthy, states=[healthy, with_condition])
    base_config.update(
        {"population": {"population_size": 1000, "age_start": 0, "age_end": 5}},
        **metadata(__file__),
    )
    simulation = InteractiveContext(
        components=[TestPopulation(), model],
        configuration=base_config,
        plugin_configuration=base_plugins,
    )

    # prevalence should be used for assigning initial status at sim start
    assert np.isclose(get_test_prevalence(simulation, "with_condition"), 1)

    # with no birth prevalence provided, it should default to 0 for ages start = end = 0
    simulation.step()
    simulation.simulant_creator(
        1000,
        population_configuration={"age_start": 0, "age_end": 0, "sim_state": "time_step"},
    )
    assert np.isclose(get_test_prevalence(simulation, "with_condition"), 0.5, 0.01)

    # and default to prevalence for ages not start = end = 0
    simulation.step()
    simulation.simulant_creator(
        1000,
        population_configuration={"age_start": 0, "age_end": 5, "sim_state": "time_step"},
    )
    assert np.isclose(get_test_prevalence(simulation, "with_condition"), 0.67, 0.01)


def test_state_transition_names(disease):
    with_condition = DiseaseState("diarrheal_diseases")
    healthy = SusceptibleState("diarrheal_diseases")
    healthy.add_rate_transition(with_condition)
    with_condition.add_rate_transition(healthy)
    model = DiseaseModel(disease, initial_state=healthy, states=[healthy, with_condition])
    assert set(model.state_names) == {
        "diarrheal_diseases",
        "susceptible_to_diarrheal_diseases",
    }
    assert set(model.transition_names) == {
        TransitionString("diarrheal_diseases_TO_susceptible_to_diarrheal_diseases"),
        TransitionString("susceptible_to_diarrheal_diseases_TO_diarrheal_diseases"),
    }


def test_artifact_transition_keys(mocker, disease):
    """Test that we use expected artifact keys to load transition data."""
    builder = mocker.Mock()
    builder.data.load = mocker.Mock()
    cause = "diarrheal_diseases"
    with_condition = DiseaseState(cause)
    healthy = SusceptibleState(cause)

    # check incidence rate
    healthy.add_rate_transition(with_condition)
    incident_transition = healthy.transition_set.transitions[0]
    incident_transition.load_transition_rate(builder)
    builder.data.load.assert_called_with(f"cause.{cause}.incidence_rate")

    # check remission rate
    with_condition.add_rate_transition(healthy)
    remissive_transition = with_condition.transition_set.transitions[0]
    remissive_transition.load_transition_rate(builder)
    builder.data.load.assert_called_with(f"cause.{cause}.remission_rate")
