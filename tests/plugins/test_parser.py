from typing import Dict, List

import pytest
from vivarium import Component, ConfigTree, InteractiveContext
from vivarium.framework.state_machine import Transient, Transition

from vivarium_public_health.disease import (
    BaseDiseaseState,
    DiseaseModel,
    DiseaseState,
    ProportionTransition,
    RateTransition,
    RecoveredState,
    SusceptibleState,
    TransientDiseaseState,
)
from vivarium_public_health.plugins import CausesConfigurationParser
from vivarium_public_health.testing.mock_artifact import MockArtifact
from vivarium_public_health.testing.mock_artifact import (
    MockArtifactManager as MockArtifactManager_,
)

SIMPLE_SIR_MODEL = "simple_sir_model"
SUSCEPTIBLE_STATE = "susceptible"
SIMPLE_MODEL_INFECTED_STATE = "some_infected_state_name"
RECOVERED_STATE = "recovered"

COMPLEX_MODEL = "complex_model"
TRANSIENT_STATE = "some_transient_state_name"
COMPLEX_MODEL_INFECTED_STATE_1 = "some_other_infected_state_name"
COMPLEX_MODEL_INFECTED_STATE_2 = "yet_another_infected_state_name"
COMPLEX_MODEL_INFECTED_STATE_3 = "still_another_infected_state_name"


PREVALENCE_DATA_FROM_FUNCTION = 0.08


def some_prevalence_function(_, __):
    return PREVALENCE_DATA_FROM_FUNCTION


RATE_DATA_FROM_FUNCTION = 0.95


def some_rate_function(_, __, ___):
    return RATE_DATA_FROM_FUNCTION


REMISSION_DATA_FROM_FUNCTION = 0.85


def some_remission_function(_, __, ___):
    return REMISSION_DATA_FROM_FUNCTION


class MockArtifactManager(MockArtifactManager_):
    def _load_artifact(self, _: str) -> MockArtifact:
        artifact = MockArtifact()

        artifact.mocks[f"cause.{SIMPLE_MODEL_INFECTED_STATE}.prevalence"] = 0.11
        artifact.mocks[f"cause.{SIMPLE_MODEL_INFECTED_STATE}.disability_weight"] = 0.12
        artifact.mocks[f"cause.{SIMPLE_MODEL_INFECTED_STATE}.excess_mortality_rate"] = 0.13

        artifact.mocks["cause.some_custom_cause.disability_weight"] = 0.22
        artifact.mocks["cause.some_custom_cause.excess_mortality_rate"] = 0.23

        artifact.mocks[f"cause.{COMPLEX_MODEL_INFECTED_STATE_2}.prevalence"] = 0.31
        artifact.mocks[f"cause.{COMPLEX_MODEL_INFECTED_STATE_2}.disability_weight"] = 0.32
        artifact.mocks[f"cause.{COMPLEX_MODEL_INFECTED_STATE_2}.excess_mortality_rate"] = 0.33

        artifact.mocks[f"cause.{COMPLEX_MODEL_INFECTED_STATE_3}.prevalence"] = 0.41
        artifact.mocks[f"cause.{COMPLEX_MODEL_INFECTED_STATE_3}.disability_weight"] = 0.42
        artifact.mocks[f"cause.{COMPLEX_MODEL_INFECTED_STATE_3}.excess_mortality_rate"] = 0.43

        return artifact


# @pytest.fixture(scope="module")
def get_component_config() -> ConfigTree:
    component_config = {
        "causes": {
            SIMPLE_SIR_MODEL: {
                "states": {
                    SUSCEPTIBLE_STATE: {},
                    SIMPLE_MODEL_INFECTED_STATE: {},
                    RECOVERED_STATE: {},
                },
                "transitions": {
                    "infected_incidence": {
                        "source": SUSCEPTIBLE_STATE,
                        "sink": SIMPLE_MODEL_INFECTED_STATE,
                        "data_type": "rate",
                        "data_sources": {"incidence_rate": 0.5},
                    },
                    "infected_remission": {
                        "source": SIMPLE_MODEL_INFECTED_STATE,
                        "sink": RECOVERED_STATE,
                        "data_type": "rate",
                        "data_sources": {"remission_rate": 0.6},
                    },
                },
            },
            COMPLEX_MODEL: {
                "states": {
                    SUSCEPTIBLE_STATE: {},
                    COMPLEX_MODEL_INFECTED_STATE_1: {
                        "cause_type": "sequela",
                        "allow_self_transition": False,
                        "data_sources": {
                            "prevalence": "tests.plugins.test_parser::some_prevalence_function",
                            "disability_weight": "cause.some_custom_cause.disability_weight",
                            "excess_mortality_rate": "cause.some_custom_cause.excess_mortality_rate",
                        },
                    },
                    TRANSIENT_STATE: {"transient": True},
                    COMPLEX_MODEL_INFECTED_STATE_2: {
                        "data_sources": {
                            "birth_prevalence": 0.3,
                            "dwell_time": "3 days",
                        },
                    },
                    COMPLEX_MODEL_INFECTED_STATE_3: {},
                },
                "transitions": {
                    "infected_state_1_incidence": {
                        "source": SUSCEPTIBLE_STATE,
                        "sink": COMPLEX_MODEL_INFECTED_STATE_1,
                        "data_type": "rate",
                        "data_sources": {"incidence_rate": 0.2},
                    },
                    "infected_state_1_to_transient": {
                        "source": COMPLEX_MODEL_INFECTED_STATE_1,
                        "sink": TRANSIENT_STATE,
                        "data_type": "proportion",
                        "data_sources": {"proportion": 0.25},
                    },
                    "infected_state_1_to_infected_state_2": {
                        "source": COMPLEX_MODEL_INFECTED_STATE_1,
                        "sink": COMPLEX_MODEL_INFECTED_STATE_2,
                        "data_type": "proportion",
                        "data_sources": {"proportion": 0.75},
                    },
                    "transient_to_infected_state_2": {
                        "source": TRANSIENT_STATE,
                        "sink": COMPLEX_MODEL_INFECTED_STATE_2,
                        "data_type": "proportion",
                        "data_sources": {"proportion": 1.0},
                    },
                    "infected_state_2_to_infected_state_3": {
                        "source": COMPLEX_MODEL_INFECTED_STATE_2,
                        "sink": COMPLEX_MODEL_INFECTED_STATE_3,
                        "data_type": "dwell_time",
                    },
                    "infected_state_3_to_infected_state_1": {
                        "source": COMPLEX_MODEL_INFECTED_STATE_3,
                        "sink": COMPLEX_MODEL_INFECTED_STATE_1,
                        "data_type": "rate",
                        "data_sources": {
                            "transition_rate": "tests.plugins.test_parser::some_rate_function"
                        },
                    },
                    "infected_state_3_remission": {
                        "source": COMPLEX_MODEL_INFECTED_STATE_3,
                        "sink": SUSCEPTIBLE_STATE,
                        "data_type": "rate",
                        "data_sources": {
                            "transition_rate": "tests.plugins.test_parser::some_remission_function"
                        },
                    },
                },
            },
        },
        "vivarium": {"testing_utilities": "TestPopulation()"},
    }
    return ConfigTree(
        component_config,
        layers=[
            "base",
            "user_configs",
            "component_configs",
            "model_override",
            "override",
        ],
    )


@pytest.fixture(scope="module")
def base_config(base_config_factory) -> ConfigTree:
    yield base_config_factory()


@pytest.fixture(scope="module")
def causes_config_parser_plugins() -> ConfigTree:
    config_parser_plugin_config = {
        "required": {
            "data": {
                "controller": "tests.plugins.test_parser.MockArtifactManager",
                "builder_interface": "vivarium.framework.artifact.ArtifactInterface",
            },
            "component_configuration_parser": {
                "controller": "vivarium_public_health.plugins.CausesConfigurationParser",
            },
        }
    }
    return ConfigTree(config_parser_plugin_config)


@pytest.fixture(scope="module")
def sim_components(base_config: ConfigTree, causes_config_parser_plugins: ConfigTree):
    simulation = InteractiveContext(
        components=get_component_config(),
        configuration=base_config,
        plugin_configuration=causes_config_parser_plugins,
    )
    return simulation.list_components()


@pytest.fixture(scope="module")
def component_list_parsed_from_config() -> List[Component]:
    config_parser = CausesConfigurationParser()
    component_config = get_component_config()
    return config_parser.parse_component_config(component_config)


def _get_transitions_from_state(state: BaseDiseaseState) -> Dict[str, Transition]:
    return {
        transition.output_state.state_id: transition
        for transition in state.transition_set.transitions
    }


def test_parser_returns_list_of_components(component_list_parsed_from_config):
    assert isinstance(component_list_parsed_from_config, list)
    expected_component_names = {
        f"disease_model.{SIMPLE_SIR_MODEL}",
        f"disease_model.{COMPLEX_MODEL}",
        "test_population",
    }
    assert expected_component_names == {
        component.name for component in component_list_parsed_from_config
    }


def test_simple_sir_disease_model(sim_components):
    sir_model = sim_components[f"disease_model.{SIMPLE_SIR_MODEL}"]
    assert isinstance(sir_model, DiseaseModel)

    # the disease model's states have the expected names
    expected_state_names = {
        f"susceptible_state.susceptible_to_{SIMPLE_SIR_MODEL}",
        f"disease_state.{SIMPLE_MODEL_INFECTED_STATE}",
        f"recovered_state.recovered_from_{SIMPLE_SIR_MODEL}",
    }
    actual_state_names = {state.name for state in sir_model.sub_components}
    assert actual_state_names == expected_state_names


def test_sir_model_susceptible_state(sim_components):
    susceptible_state = sim_components[f"susceptible_state.susceptible_to_{SIMPLE_SIR_MODEL}"]
    assert isinstance(susceptible_state, SusceptibleState)
    assert susceptible_state.state_id == f"susceptible_to_{SIMPLE_SIR_MODEL}"

    # test that it has the expected default values
    assert susceptible_state.cause_type == "cause"
    assert not isinstance(susceptible_state, Transient)
    assert susceptible_state.transition_set.allow_null_transition

    # test that it has the expected transition
    transitions = _get_transitions_from_state(susceptible_state)
    assert set(transitions.keys()) == {SIMPLE_MODEL_INFECTED_STATE}

    incidence_transition = transitions[SIMPLE_MODEL_INFECTED_STATE]
    assert isinstance(incidence_transition, RateTransition)
    assert incidence_transition.base_rate.data == 0.5


def test_sir_model_disease_state(sim_components):
    infected_state = sim_components[f"disease_state.{SIMPLE_MODEL_INFECTED_STATE}"]
    assert isinstance(infected_state, DiseaseState)
    assert infected_state.state_id == SIMPLE_MODEL_INFECTED_STATE

    # test that it has the expected default values
    assert infected_state.cause_type == "cause"
    assert not isinstance(infected_state, Transient)
    assert infected_state.transition_set.allow_null_transition

    # test we get the default data sources
    assert infected_state.prevalence.data == 0.11
    assert infected_state.birth_prevalence.data == 0.0
    assert infected_state.dwell_time.source.data == 0.0
    assert infected_state.base_disability_weight.data == 0.12
    assert infected_state.base_excess_mortality_rate.data == 0.13

    # test that it has the expected transition
    transitions = _get_transitions_from_state(infected_state)
    assert set(transitions.keys()) == {f"recovered_from_{SIMPLE_SIR_MODEL}"}
    incidence_transition = transitions[f"recovered_from_{SIMPLE_SIR_MODEL}"]
    assert isinstance(incidence_transition, RateTransition)
    assert incidence_transition.base_rate.data == 0.6


def test_sir_recovered_recovered_state(sim_components):
    recovered_state = sim_components[f"recovered_state.recovered_from_{SIMPLE_SIR_MODEL}"]
    assert isinstance(recovered_state, RecoveredState)
    assert recovered_state.state_id == f"recovered_from_{SIMPLE_SIR_MODEL}"

    # test that it has the expected default values
    assert recovered_state.cause_type == "cause"
    assert not isinstance(recovered_state, Transient)
    assert recovered_state.transition_set.allow_null_transition

    # test that it has the expected transitions
    assert len(recovered_state.transition_set.transitions) == 0


# todo test config file references external config file
# todo test config file defines causes itself
# todo test config input as dict
# todo test config file with both local and external definition of causes


def test_complex_model(sim_components):
    complex_model = sim_components[f"disease_model.{COMPLEX_MODEL}"]
    assert isinstance(complex_model, DiseaseModel)

    # the disease model's states have the expected names
    expected_state_names = {
        f"susceptible_state.susceptible_to_{COMPLEX_MODEL}",
        f"disease_state.{COMPLEX_MODEL_INFECTED_STATE_1}",
        f"disease_state.{COMPLEX_MODEL_INFECTED_STATE_2}",
        f"disease_state.{COMPLEX_MODEL_INFECTED_STATE_3}",
        f"transient_disease_state.{TRANSIENT_STATE}",
    }
    actual_state_names = {state.name for state in complex_model.sub_components}
    assert actual_state_names == expected_state_names


def test_complex_model_first_disease_state(sim_components):
    infected_state = sim_components[f"disease_state.{COMPLEX_MODEL_INFECTED_STATE_1}"]
    assert isinstance(infected_state, DiseaseState)
    assert infected_state.state_id == COMPLEX_MODEL_INFECTED_STATE_1

    # test that it has the expected default and configured values
    assert infected_state.cause_type == "sequela"
    assert not isinstance(infected_state, Transient)
    assert not infected_state.transition_set.allow_null_transition

    # test we get the expected default and configured data sources
    assert infected_state.prevalence.data == PREVALENCE_DATA_FROM_FUNCTION
    assert infected_state.birth_prevalence.data == 0.0
    assert infected_state.dwell_time.source.data == 0.0
    assert infected_state.base_disability_weight.data == 0.22
    assert infected_state.base_excess_mortality_rate.data == 0.23

    # test that it has the expected transitions
    transitions = _get_transitions_from_state(infected_state)
    assert set(transitions.keys()) == {COMPLEX_MODEL_INFECTED_STATE_2, TRANSIENT_STATE}

    to_transient_transition = transitions[TRANSIENT_STATE]
    assert isinstance(to_transient_transition, ProportionTransition)
    assert to_transient_transition.proportion.data == 0.25

    to_infected_2_transition = transitions[COMPLEX_MODEL_INFECTED_STATE_2]
    assert isinstance(to_infected_2_transition, ProportionTransition)
    assert to_infected_2_transition.proportion.data == 0.75


def test_complex_model_transient_disease_state(sim_components):
    transient_state = sim_components[f"transient_disease_state.{TRANSIENT_STATE}"]
    assert isinstance(transient_state, TransientDiseaseState)
    assert transient_state.state_id == TRANSIENT_STATE

    # test that it has the expected default and configured values
    assert transient_state.cause_type == "cause"
    assert isinstance(transient_state, Transient)

    # test that it has the expected transitions
    transitions = _get_transitions_from_state(transient_state)
    assert set(transitions.keys()) == {COMPLEX_MODEL_INFECTED_STATE_2}

    transition = transitions[COMPLEX_MODEL_INFECTED_STATE_2]
    assert isinstance(transition, ProportionTransition)
    assert transition.proportion.data == 1.0


def test_complex_model_second_disease_state(sim_components):
    infected_state = sim_components[f"disease_state.{COMPLEX_MODEL_INFECTED_STATE_2}"]
    assert isinstance(infected_state, DiseaseState)
    assert infected_state.state_id == COMPLEX_MODEL_INFECTED_STATE_2

    # test that it has the expected default and configured values
    assert infected_state.cause_type == "cause"
    assert not isinstance(infected_state, Transient)
    assert infected_state.transition_set.allow_null_transition

    # test we get the expected default and configured data sources
    assert infected_state.prevalence.data == 0.31
    assert infected_state.birth_prevalence.data == 0.3
    assert infected_state.dwell_time.source.data == 3.0
    assert infected_state.base_disability_weight.data == 0.32
    assert infected_state.base_excess_mortality_rate.data == 0.33

    # test that it has the expected transitions
    transitions = _get_transitions_from_state(infected_state)
    assert set(transitions.keys()) == {COMPLEX_MODEL_INFECTED_STATE_3}

    transition = transitions[COMPLEX_MODEL_INFECTED_STATE_3]
    assert isinstance(transition, Transition)
    assert not isinstance(transition, ProportionTransition)
    assert not isinstance(transition, RateTransition)


def test_complex_model_third_disease_state(sim_components):
    infected_state = sim_components[f"disease_state.{COMPLEX_MODEL_INFECTED_STATE_3}"]
    assert isinstance(infected_state, DiseaseState)
    assert infected_state.state_id == COMPLEX_MODEL_INFECTED_STATE_3

    # test that it has the expected default values
    assert infected_state.cause_type == "cause"
    assert not isinstance(infected_state, Transient)
    assert infected_state.transition_set.allow_null_transition

    # test we get the default data sources
    assert infected_state.prevalence.data == 0.41
    assert infected_state.birth_prevalence.data == 0.0
    assert infected_state.dwell_time.source.data == 0.0
    assert infected_state.base_disability_weight.data == 0.42
    assert infected_state.base_excess_mortality_rate.data == 0.43

    # test that it has the expected transition
    transitions = _get_transitions_from_state(infected_state)
    assert set(transitions.keys()) == {
        COMPLEX_MODEL_INFECTED_STATE_1,
        f"susceptible_to_{COMPLEX_MODEL}",
    }

    to_infected_1_transition = transitions[COMPLEX_MODEL_INFECTED_STATE_1]
    assert isinstance(to_infected_1_transition, RateTransition)
    assert to_infected_1_transition.base_rate.data == RATE_DATA_FROM_FUNCTION

    to_susceptible_transition = transitions[f"susceptible_to_{COMPLEX_MODEL}"]
    assert isinstance(to_susceptible_transition, RateTransition)
    assert to_susceptible_transition.base_rate.data == REMISSION_DATA_FROM_FUNCTION


# todo test invalid data source
