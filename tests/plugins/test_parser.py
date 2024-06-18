import dataclasses
from typing import Any, Dict, List, NamedTuple, Tuple, Type

import numpy as np
import pytest
import yaml
from layered_config_tree import LayeredConfigTree
from vivarium import Component, InteractiveContext
from vivarium.framework.components.parser import ParsingError
from vivarium.framework.state_machine import Transient, Transition

from tests.mock_artifact import MockArtifact
from tests.mock_artifact import MockArtifactManager as MockArtifactManager_
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

SIR_MODEL = "simple_sir_model"
SIR_SUSCEPTIBLE_NAME = "susceptible_to_simple_sir_model"
SIR_INFECTED_STATE_NAME = "sir_infected_state_name"
SIR_RECOVERED_NAME = "recovered_from_simple_sir_model"

COMPLEX_MODEL = "complex_model"
COMPLEX_STATE_1_NAME = "complex_infected_state_name"
TRANSIENT_STATE_NAME = "some_transient_state_name"
COMPLEX_STATE_2_NAME = "another_complex_infected_state_name"
COMPLEX_STATE_3_NAME = "yet_another_complex_infected_state_name"


class ComplexModel(DiseaseModel):
    pass


class ComplexState(DiseaseState):
    pass


SIR_MODEL_CSMR = 0.9
COMPLEX_MODEL_CSMR = 1.4


@dataclasses.dataclass
class ExpectedTransitionData:
    source: str
    sink: str
    transition_type: Type[Transition]
    value: float


@dataclasses.dataclass
class ExpectedStateData:
    name: str
    state_type: Type[BaseDiseaseState] = DiseaseState
    cause_type: str = "cause"
    is_transient: bool = False
    allow_self_transition: bool = True
    prevalence: float = 0.0
    birth_prevalence: float = 0.0
    dwell_time: float = 0.0
    disability_weight: float = 0.0
    emr: float = 0.0

    transitions: List[ExpectedTransitionData] = dataclasses.field(default_factory=list)

    def get_transitions(self) -> Dict[str, ExpectedTransitionData]:
        """Return a dict of transitions keyed by their sink state name."""
        return {transition.sink: transition for transition in self.transitions}


class ExpectedStates(NamedTuple):
    SIR_SUSCEPTIBLE: ExpectedStateData = ExpectedStateData(
        name=SIR_SUSCEPTIBLE_NAME,
        state_type=SusceptibleState,
        transitions=[
            ExpectedTransitionData(
                SIR_SUSCEPTIBLE_NAME, SIR_INFECTED_STATE_NAME, RateTransition, 0.5
            )
        ],
    )
    SIR_INFECTED: ExpectedStateData = ExpectedStateData(
        name=SIR_INFECTED_STATE_NAME,
        prevalence=0.11,
        disability_weight=0.12,
        emr=0.13,
        transitions=[
            ExpectedTransitionData(
                SIR_INFECTED_STATE_NAME, SIR_RECOVERED_NAME, RateTransition, 0.6
            )
        ],
    )
    SIR_RECOVERED: ExpectedStateData = ExpectedStateData(
        name=SIR_RECOVERED_NAME, state_type=RecoveredState
    )

    COMPLEX_INFECTED_1: ExpectedStateData = ExpectedStateData(
        name=COMPLEX_STATE_1_NAME,
        cause_type="sequela",
        allow_self_transition=False,
        prevalence=0.21,
        disability_weight=0.22,
        emr=0.23,
        transitions=[
            ExpectedTransitionData(
                COMPLEX_STATE_1_NAME,
                TRANSIENT_STATE_NAME,
                ProportionTransition,
                0.25,
            ),
            ExpectedTransitionData(
                COMPLEX_STATE_1_NAME,
                COMPLEX_STATE_2_NAME,
                ProportionTransition,
                0.75,
            ),
        ],
    )
    TRANSIENT: ExpectedStateData = ExpectedStateData(
        name=TRANSIENT_STATE_NAME,
        state_type=TransientDiseaseState,
        is_transient=True,
        transitions=[
            ExpectedTransitionData(
                TRANSIENT_STATE_NAME, COMPLEX_STATE_2_NAME, ProportionTransition, 1.0
            ),
        ],
    )
    COMPLEX_INFECTED_2: ExpectedStateData = ExpectedStateData(
        name=COMPLEX_STATE_2_NAME,
        state_type=ComplexState,
        prevalence=0.31,
        birth_prevalence=0.3,
        dwell_time=3.0,
        disability_weight=0.32,
        emr=0.33,
        transitions=[
            ExpectedTransitionData(
                COMPLEX_STATE_2_NAME, COMPLEX_STATE_3_NAME, Transition, 0.0
            ),
        ],
    )

    COMPLEX_INFECTED_3: ExpectedStateData = ExpectedStateData(
        name=COMPLEX_STATE_3_NAME,
        prevalence=0.41,
        disability_weight=0.42,
        emr=0.43,
        transitions=[
            ExpectedTransitionData(
                COMPLEX_STATE_3_NAME, COMPLEX_STATE_1_NAME, RateTransition, 0.95
            ),
            ExpectedTransitionData(
                COMPLEX_STATE_3_NAME, COMPLEX_STATE_2_NAME, RateTransition, 0.85
            ),
        ],
    )


STATES = ExpectedStates()


def complex_model_infected_1_prevalence(_, __):
    return STATES.COMPLEX_INFECTED_1.prevalence


def complex_model_3_to_1_transition_rate(_, __, ___):
    return STATES.COMPLEX_INFECTED_3.get_transitions()[COMPLEX_STATE_1_NAME].value


def complex_model_3_to_2_transition_rate(_, __, ___):
    return STATES.COMPLEX_INFECTED_3.get_transitions()[COMPLEX_STATE_2_NAME].value


class MockArtifactManager(MockArtifactManager_):
    def _load_artifact(self, _: str) -> MockArtifact:
        artifact = MockArtifact()

        artifact.mocks[f"cause.{SIR_MODEL}.cause_specific_mortality_rate"] = SIR_MODEL_CSMR
        artifact.mocks[
            f"cause.{STATES.SIR_INFECTED.name}.prevalence"
        ] = STATES.SIR_INFECTED.prevalence
        artifact.mocks[
            f"cause.{STATES.SIR_INFECTED.name}.disability_weight"
        ] = STATES.SIR_INFECTED.disability_weight
        artifact.mocks[
            f"cause.{STATES.SIR_INFECTED.name}.excess_mortality_rate"
        ] = STATES.SIR_INFECTED.emr

        artifact.mocks[
            "cause.some_custom_cause.disability_weight"
        ] = STATES.COMPLEX_INFECTED_1.disability_weight
        artifact.mocks[
            "cause.some_custom_cause.excess_mortality_rate"
        ] = STATES.COMPLEX_INFECTED_1.emr

        artifact.mocks[
            f"cause.{STATES.COMPLEX_INFECTED_2.name}.prevalence"
        ] = STATES.COMPLEX_INFECTED_2.prevalence
        artifact.mocks[
            f"cause.{STATES.COMPLEX_INFECTED_2.name}.disability_weight"
        ] = STATES.COMPLEX_INFECTED_2.disability_weight
        artifact.mocks[
            f"cause.{STATES.COMPLEX_INFECTED_2.name}.excess_mortality_rate"
        ] = STATES.COMPLEX_INFECTED_2.emr

        artifact.mocks[
            f"cause.{STATES.COMPLEX_INFECTED_3.name}.prevalence"
        ] = STATES.COMPLEX_INFECTED_3.prevalence
        artifact.mocks[
            f"cause.{STATES.COMPLEX_INFECTED_3.name}.disability_weight"
        ] = STATES.COMPLEX_INFECTED_3.disability_weight
        artifact.mocks[
            f"cause.{STATES.COMPLEX_INFECTED_3.name}.excess_mortality_rate"
        ] = STATES.COMPLEX_INFECTED_3.emr

        return artifact


SIR_MODEL_CONFIG = {
    SIR_MODEL: {
        "states": {
            "susceptible": {},
            STATES.SIR_INFECTED.name: {},
            "recovered": {},
        },
        "transitions": {
            "infected_incidence": {
                "source": "susceptible",
                "sink": STATES.SIR_INFECTED.name,
                "transition_type": "rate",
                "data_sources": {
                    "incidence_rate": STATES.SIR_SUSCEPTIBLE.get_transitions()[
                        STATES.SIR_INFECTED.name
                    ].value
                },
            },
            "infected_remission": {
                "source": STATES.SIR_INFECTED.name,
                "sink": "recovered",
                "transition_type": "rate",
                "data_sources": {
                    "remission_rate": STATES.SIR_INFECTED.get_transitions()[
                        f"recovered_from_{SIR_MODEL}"
                    ].value
                },
            },
        },
    }
}


COMPLEX_MODEL_CONFIG = {
    COMPLEX_MODEL: {
        "model_type": "tests.plugins.test_parser.ComplexModel",
        "initial_state": STATES.COMPLEX_INFECTED_1.name,
        "data_sources": {"cause_specific_mortality_rate": COMPLEX_MODEL_CSMR},
        "states": {
            STATES.COMPLEX_INFECTED_1.name: {
                "cause_type": STATES.COMPLEX_INFECTED_1.cause_type,
                "allow_self_transition": STATES.COMPLEX_INFECTED_1.allow_self_transition,
                "data_sources": {
                    "prevalence": "tests.plugins.test_parser::complex_model_infected_1_prevalence",
                    "disability_weight": "cause.some_custom_cause.disability_weight",
                    "excess_mortality_rate": "cause.some_custom_cause.excess_mortality_rate",
                },
            },
            TRANSIENT_STATE_NAME: {"transient": True},
            COMPLEX_STATE_2_NAME: {
                "state_type": "tests.plugins.test_parser.ComplexState",
                "data_sources": {
                    "birth_prevalence": STATES.COMPLEX_INFECTED_2.birth_prevalence,
                    "dwell_time": f"{STATES.COMPLEX_INFECTED_2.dwell_time} days",
                },
            },
            COMPLEX_STATE_3_NAME: {},
        },
        "transitions": {
            "infected_state_1_to_transient": {
                "source": STATES.COMPLEX_INFECTED_1.name,
                "sink": TRANSIENT_STATE_NAME,
                "transition_type": "proportion",
                "data_sources": {
                    "proportion": STATES.COMPLEX_INFECTED_1.get_transitions()[
                        TRANSIENT_STATE_NAME
                    ].value
                },
            },
            "infected_state_1_to_infected_state_2": {
                "source": STATES.COMPLEX_INFECTED_1.name,
                "sink": COMPLEX_STATE_2_NAME,
                "transition_type": "proportion",
                "data_sources": {
                    "proportion": STATES.COMPLEX_INFECTED_1.get_transitions()[
                        COMPLEX_STATE_2_NAME
                    ].value
                },
            },
            "transient_to_infected_state_2": {
                "source": TRANSIENT_STATE_NAME,
                "sink": COMPLEX_STATE_2_NAME,
                "transition_type": "proportion",
                "data_sources": {
                    "proportion": STATES.TRANSIENT.get_transitions()[
                        COMPLEX_STATE_2_NAME
                    ].value
                },
            },
            "infected_state_2_to_infected_state_3": {
                "source": COMPLEX_STATE_2_NAME,
                "sink": COMPLEX_STATE_3_NAME,
                "transition_type": "dwell_time",
            },
            "infected_state_3_to_infected_state_1": {
                "source": COMPLEX_STATE_3_NAME,
                "sink": STATES.COMPLEX_INFECTED_1.name,
                "transition_type": "rate",
                "data_sources": {
                    "transition_rate": "tests.plugins.test_parser::complex_model_3_to_1_transition_rate"
                },
            },
            "infected_state_3_to_infected_state_2": {
                "source": COMPLEX_STATE_3_NAME,
                "sink": STATES.COMPLEX_INFECTED_2.name,
                "transition_type": "rate",
                "data_sources": {
                    "transition_rate": "tests.plugins.test_parser::complex_model_3_to_2_transition_rate"
                },
            },
        },
    }
}


def create_simulation_config_tree(config_dict: Dict) -> LayeredConfigTree:
    config_tree_layers = [
        "base",
        "user_configs",
        "component_configs",
        "model_override",
        "override",
    ]
    config_tree = LayeredConfigTree(layers=config_tree_layers)
    config_tree.update(config_dict, layer="model_override")
    return config_tree


@pytest.fixture(scope="module")
def base_config(base_config_factory) -> LayeredConfigTree:
    yield base_config_factory()


@pytest.fixture(scope="module")
def causes_config_parser_plugins() -> LayeredConfigTree:
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
    return LayeredConfigTree(config_parser_plugin_config)


@pytest.fixture
def resource_filename_mock(tmp_path, mocker):
    resource_filename_mock = mocker.patch(
        "vivarium_public_health.plugins.parser.resource_filename"
    )
    resource_filename_mock.side_effect = lambda _, filename: str(tmp_path / filename)
    return resource_filename_mock


ALL_COMPONENTS_CONFIG_DICT = {
    "causes": {**SIR_MODEL_CONFIG, **COMPLEX_MODEL_CONFIG},
    "vivarium": {"testing_utilities": "TestPopulation()"},
}


@pytest.fixture(scope="module")
def sim_components(
    base_config: LayeredConfigTree, causes_config_parser_plugins: LayeredConfigTree
):
    simulation = InteractiveContext(
        components=create_simulation_config_tree(ALL_COMPONENTS_CONFIG_DICT),
        configuration=base_config,
        plugin_configuration=causes_config_parser_plugins,
    )
    return simulation.list_components()


##############################
# Test configuration parsing #
##############################


def _test_parsing_of_config_file(
    component_config: LayeredConfigTree,
    expected_component_names: Tuple[str] = (
        f"disease_model.{SIR_MODEL}",
        f"complex_model.{COMPLEX_MODEL}",
        "test_population",
    ),
):
    parsed_components = CausesConfigurationParser().parse_component_config(component_config)
    assert isinstance(parsed_components, list)
    assert len(parsed_components) == len(expected_component_names)
    for component in parsed_components:
        assert component.name in expected_component_names


def test_parser_returns_list_of_components():
    config = create_simulation_config_tree(ALL_COMPONENTS_CONFIG_DICT)
    _test_parsing_of_config_file(config)


def test_parsing_config_single_external_causes_config_file(tmp_path, resource_filename_mock):
    causes_config = {"causes": {**SIR_MODEL_CONFIG, **COMPLEX_MODEL_CONFIG}}
    with open(tmp_path / "causes_config.yaml", "w") as file:
        yaml.dump(causes_config, file)

    component_config = create_simulation_config_tree(
        {
            "external_configuration": {"some_repo": ["causes_config.yaml"]},
            "vivarium": {"testing_utilities": "TestPopulation()"},
        }
    )
    _test_parsing_of_config_file(component_config)


def test_parsing_config_multiple_external_causes_config_file(
    tmp_path, resource_filename_mock
):
    with open(tmp_path / "sir.yaml", "w") as file:
        yaml.dump({"causes": SIR_MODEL_CONFIG}, file)

    with open(tmp_path / "complex.yaml", "w") as file:
        yaml.dump({"causes": COMPLEX_MODEL_CONFIG}, file)

    component_config = create_simulation_config_tree(
        {
            "external_configuration": {"some_repo": ["sir.yaml", "complex.yaml"]},
            "vivarium": {"testing_utilities": "TestPopulation()"},
        }
    )
    _test_parsing_of_config_file(component_config)


def test_parsing_config_external_and_local_causes_config_file(
    tmp_path, resource_filename_mock
):
    with open(tmp_path / "sir.yaml", "w") as file:
        yaml.dump({"causes": SIR_MODEL_CONFIG}, file)

    component_config = create_simulation_config_tree(
        {
            "external_configuration": {"some_repo": ["sir.yaml"]},
            "causes": COMPLEX_MODEL_CONFIG,
            "vivarium": {"testing_utilities": "TestPopulation()"},
        }
    )

    _test_parsing_of_config_file(component_config)


def test_parsing_no_causes_config_file(tmp_path, resource_filename_mock):
    component_config = create_simulation_config_tree(
        {"vivarium": {"testing_utilities": "TestPopulation()"}}
    )
    _test_parsing_of_config_file(
        component_config, expected_component_names=("test_population",)
    )


@pytest.mark.parametrize(
    "config_dict, expected_error_message",
    [
        ({3: ["sir.yaml"]}, "must be a string definition of a package"),
        ({"some_repo": 3}, "must be a list"),
        ({"some_repo": ["sir.yaml", 3]}, "paths to yaml files"),
        ({"some_repo": ["sir.yaml", "complex.yaml", "bad"]}, "paths to yaml files"),
    ],
)
def test_parsing_invalid_external_configuration(config_dict, expected_error_message):
    component_config = create_simulation_config_tree(
        {
            "external_configuration": config_dict,
            "vivarium": {"testing_utilities": "TestPopulation()"},
        }
    )
    with pytest.raises(ParsingError, match=expected_error_message):
        CausesConfigurationParser().parse_component_config(component_config)


@pytest.mark.parametrize(
    "model_name, expected_model_type, expected_csmr, expected_initial_state, "
    "expected_state_names",
    [
        (
            f"disease_model.{SIR_MODEL}",
            DiseaseModel,
            SIR_MODEL_CSMR,
            STATES.SIR_SUSCEPTIBLE.name,
            [
                f"susceptible_state.{STATES.SIR_SUSCEPTIBLE.name}",
                f"disease_state.{STATES.SIR_INFECTED.name}",
                f"recovered_state.{STATES.SIR_RECOVERED.name}",
            ],
        ),
        (
            f"complex_model.{COMPLEX_MODEL}",
            ComplexModel,
            COMPLEX_MODEL_CSMR,
            STATES.COMPLEX_INFECTED_1.name,
            [
                f"disease_state.{STATES.COMPLEX_INFECTED_1.name}",
                f"complex_state.{STATES.COMPLEX_INFECTED_2.name}",
                f"disease_state.{STATES.COMPLEX_INFECTED_3.name}",
                f"transient_disease_state.{STATES.TRANSIENT.name}",
            ],
        ),
    ],
)
def test_disease_model(
    sim_components: Dict[str, Component],
    model_name: str,
    expected_csmr: float,
    expected_model_type: Type[DiseaseModel],
    expected_initial_state: str,
    expected_state_names: List[str],
):
    model = sim_components[model_name]
    assert isinstance(model, expected_model_type)
    assert model.initial_state == expected_initial_state

    assert model.lookup_tables["cause_specific_mortality_rate"].data == expected_csmr

    # the disease model's states have the expected names
    actual_state_names = {state.name for state in model.sub_components}
    assert actual_state_names == set(expected_state_names)


def test_no_extra_state_components(sim_components: Dict[str, Component]):
    actual_state_names = {
        component.state_id
        for component in sim_components.values()
        if isinstance(component, BaseDiseaseState)
    }
    expected_state_names = {state.name for state in STATES}
    assert actual_state_names == expected_state_names


@pytest.mark.parametrize("expected_state_data", STATES, ids=[state.name for state in STATES])
def test_disease_state(
    sim_components: Dict[str, Component], expected_state_data: ExpectedStateData
):
    name_prefix = {
        DiseaseState: "disease_state",
        SusceptibleState: "susceptible_state",
        TransientDiseaseState: "transient_disease_state",
        RecoveredState: "recovered_state",
        BaseDiseaseState: "base_disease_state",
        ComplexState: "complex_state",
    }[expected_state_data.state_type]

    state = sim_components[f"{name_prefix}.{expected_state_data.name}"]
    assert isinstance(state, expected_state_data.state_type)

    # test all shared expected default and configured values
    assert state.state_id == expected_state_data.name
    assert state.cause_type == expected_state_data.cause_type
    assert (
        state.transition_set.allow_null_transition
        == expected_state_data.allow_self_transition
    )

    if expected_state_data.is_transient:
        assert isinstance(state, Transient)

    if isinstance(state, DiseaseState):
        assert (
            state.transition_set.allow_null_transition
            == expected_state_data.allow_self_transition
        )

        # test we get the expected default and configured data sources
        assert state.lookup_tables["prevalence"].data == expected_state_data.prevalence
        assert (
            state.lookup_tables["birth_prevalence"].data
            == expected_state_data.birth_prevalence
        )
        assert state.lookup_tables["dwell_time"].data == expected_state_data.dwell_time
        assert (
            state.lookup_tables["disability_weight"].data
            == expected_state_data.disability_weight
        )
        assert state.lookup_tables["excess_mortality_rate"].data == expected_state_data.emr

    # test that it has the expected transitions
    for transition in state.transition_set.transitions:
        expected_transition_data = expected_state_data.get_transitions()[
            transition.output_state.state_id
        ]
        assert type(transition) == expected_transition_data.transition_type
        if isinstance(transition, RateTransition):
            actual_rate = transition.lookup_tables["transition_rate"].data
            assert actual_rate == expected_transition_data.value
        elif isinstance(transition, ProportionTransition):
            actual_proportion = transition.lookup_tables["proportion"].data
            assert actual_proportion == expected_transition_data.value


####################
# Validation Tests #
####################

INVALID_CONFIG_PARAMS = {
    "not dict": (["some", "strings"], "must be a dictionary"),
    "invalid key": ({"invalid_key": "value"}, "may only contain the following keys"),
    "model type no module": (
        {"model_type": "Model"},
        "fully qualified import path to a.*DiseaseModel",
    ),
    "model type bad module": (
        {"model_type": "some.repo.Model"},
        "fully qualified import path to a.*DiseaseModel",
    ),
    "model type bad class": (
        {"model_type": "tests.plugins.test_parser.NonModel"},
        "fully qualified import path to a.*DiseaseModel",
    ),
    "model type not disease model": (
        {"model_type": "tests.plugins.test_parser.ComplexState"},
        "fully qualified import path to a.*DiseaseModel",
    ),
    "invalid cause data sources key": (
        {"data_sources": {"not_a_valid_source": 1.0}},
        "may only contain",
    ),
    "invalid cause data sources value": (
        {"data_sources": {"cause_specific_mortality_rate": "bad value"}},
        "has an invalid data source at",
    ),
    "no states key": ({"transitions": {"s": {}}}, "must define at least one state"),
    "empty states": ({"states": {}}, "must define at least one state"),
    "states not dict": ({"states": ["s1", "s2"]}, "must be a dictionary"),
    "state_1 not dict": ({"states": {"s1": ["not", "a", "dict"]}}, "must be a dictionary"),
    "initial state not in states": (
        {"initial_state": "not_here", "states": {"s1": {}}},
        "must be present in the states",
    ),
    "invalid state key": ({"states": {"s1": {"bad_key": ""}}}, "state 's1' may only contain"),
    "susceptible state with data sources": (
        {"states": {"susceptible": {"data_sources": ""}}},
        "may only contain",
    ),
    "state type no module": (
        {"states": {"s1": {"state_type": "State"}}},
        "fully qualified import path to a.*BaseDiseaseState",
    ),
    "state type bad module": (
        {"states": {"s1": {"state_type": "some.repo.State"}}},
        "fully qualified import path to a.*BaseDiseaseState",
    ),
    "state type bad class": (
        {"states": {"s1": {"state_type": "tests.plugins.test_parser.NonState"}}},
        "fully qualified import path to a.*BaseDiseaseState",
    ),
    "state type not disease state": (
        {"states": {"s1": {"state_type": "tests.plugins.test_parser.ComplexModel"}}},
        "fully qualified import path to a.*BaseDiseaseState",
    ),
    "invalid cause type": ({"states": {"s1": {"cause_type": 3}}}, "must be a string"),
    "invalid transient": ({"states": {"s1": {"transient": 3}}}, "must be a bool"),
    "state type and susceptible": (
        {"states": {"susceptible": {"state_type": "tests.plugins.test_parser.ComplexState"}}},
        "state_type is not an allowed configuration",
    ),
    "state type and recovered": (
        {"states": {"recovered": {"state_type": "tests.plugins.test_parser.ComplexState"}}},
        "state_type is not an allowed configuration",
    ),
    "state type and transient": (
        {
            "states": {
                "s1": {
                    "state_type": "tests.plugins.test_parser.ComplexState",
                    "transient": True,
                }
            }
        },
        "state_type is not an allowed configuration",
    ),
    "transient and susceptible": (
        {"states": {"susceptible": {"transient": True}}},
        "transient is not an allowed configuration",
    ),
    "transient and recovered": (
        {"states": {"recovered": {"transient": True}}},
        "transient is not an allowed configuration",
    ),
    "invalid allow self transition": (
        {"states": {"s1": {"allow_self_transition": 3}}},
        "must be a bool",
    ),
    "state data sources not dict": (
        {"states": {"s1": {"data_sources": ""}}},
        "must be a dictionary",
    ),
    "state invalid data sources key": (
        {"states": {"s1": {"data_sources": {"bad_key": ""}}}},
        "may only contain",
    ),
    "state invalid data sources value": (
        {"states": {"s1": {"data_sources": {"prevalence": "bad_value"}}}},
        "has an invalid data source at",
    ),
    "transitions not dict": ({"transitions": ["not", "a", "dict"]}, "must be a dictionary"),
    "transition_1 not dict": (
        {"transitions": {"t1": ["not", "a", "dict"]}},
        "must be a dictionary",
    ),
    "invalid transition key": ({"transitions": {"t1": {"bad_key": ""}}}, "may only contain"),
    "missing source and sink": (
        {"transitions": {"t1": {"transition_type": "rate"}}},
        "must contain both a source and a sink",
    ),
    "missing source": (
        {"transitions": {"t1": {"sink": "s1"}}},
        "must contain both a source and a sink",
    ),
    "missing sink": (
        {"transitions": {"t1": {"source": "s1"}}},
        "must contain both a source and a sink",
    ),
    "source not in states": (
        {
            "states": {"susceptible": {}, "s2": {}},
            "transitions": {"t1": {"source": "s1", "sink": "s2"}},
        },
        "source that is present in the states",
    ),
    "sink not in states": (
        {
            "states": {"susceptible": {}, "s2": {}},
            "transitions": {"t1": {"source": "s2", "sink": "s1"}},
        },
        "sink that is present in the states",
    ),
    "missing transition type": (
        {"transitions": {"t1": {"source": "s1"}}},
        "must contain a transition type",
    ),
    "invalid transition type": (
        {
            "states": {"susceptible": {}, "s2": {}},
            "transitions": {
                "t1": {"source": "susceptible", "sink": "s2", "transition_type": "bad type"}
            },
        },
        "may only contain the following values",
    ),
    "invalid triggered value": (
        {
            "states": {"susceptible": {}, "s2": {}},
            "transitions": {
                "t1": {"source": "susceptible", "sink": "s2", "triggered": "bad_value"}
            },
        },
        "may only have one of the following values",
    ),
    "dwell time with data sources": (
        {"transitions": {"t1": {"transition_type": "dwell_time", "data_sources": {}}}},
        "may not have data sources",
    ),
    "transition data sources not dict": (
        {"transitions": {"t1": {"transition_type": "rate", "data_sources": ""}}},
        "must be a dictionary",
    ),
    "rate transition invalid data sources key": (
        {
            "transitions": {
                "t1": {"transition_type": "rate", "data_sources": {"proportion": ""}}
            }
        },
        "may only contain",
    ),
    "proportion transition invalid data sources key": (
        {
            "transitions": {
                "t1": {
                    "transition_type": "proportion",
                    "data_sources": {"transition_rate": ""},
                }
            }
        },
        "may only contain",
    ),
    "transition invalid data sources value": (
        {
            "transitions": {
                "t1": {
                    "transition_type": "rate",
                    "data_sources": {"transition_rate": "bad_value"},
                }
            }
        },
        "has an invalid data source at",
    ),
}


@pytest.mark.parametrize(
    "model_config, expected_error_message",
    INVALID_CONFIG_PARAMS.values(),
    ids=INVALID_CONFIG_PARAMS.keys(),
)
def test_invalid_model_config_throws_error(model_config: Any, expected_error_message: str):
    config = create_simulation_config_tree({"causes": {"model_name": model_config}})
    expected_error_message = f"cause 'model_name'.*{expected_error_message}"
    with pytest.raises(ParsingError, match=expected_error_message):
        CausesConfigurationParser().parse_component_config(config)


def test_multiple_errors_present_in_error_message():
    config = create_simulation_config_tree(
        {
            "causes": {
                "model_name": {
                    "model_type": "some_not_found_module.Model",
                    "states": ["not", "a", "dict"],
                    "transitions": ["not", "a", "dict"],
                }
            }
        }
    )
    expected_error_message = (
        "fully qualified import path to a.*DiseaseModel.*"
        "\n - States configuration for cause 'model_name' must be a dictionary."
        "\n - Transitions configuration for cause 'model_name' must be a dictionary "
        "if it is present."
    )
    with pytest.raises(ParsingError, match=expected_error_message):
        CausesConfigurationParser().parse_component_config(config)
