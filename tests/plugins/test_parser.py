import dataclasses
from typing import Dict, List, Set, Tuple, Type, NamedTuple

import pytest
import yaml
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

SIR_MODEL = "simple_sir_model"
SIR_SUSCEPTIBLE_NAME = "susceptible_to_simple_sir_model"
SIR_INFECTED_STATE_NAME = "sir_infected_state_name"
SIR_RECOVERED_NAME = "recovered_from_simple_sir_model"

COMPLEX_MODEL = "complex_model"
COMPLEX_SUSCEPTIBLE_NAME = "susceptible_to_complex_model"
COMPLEX_INFECTED_STATE_1_NAME = "complex_infected_state_name"
TRANSIENT_STATE_NAME = "some_transient_state_name"
COMPLEX_STATE_2_NAME = "another_complex_infected_state_name"
COMPLEX_STATE_3_NAME = "yet_another_complex_infected_state_name"


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

    COMPLEX_SUSCEPTIBLE: ExpectedStateData = ExpectedStateData(
        name=COMPLEX_SUSCEPTIBLE_NAME,
        state_type=SusceptibleState,
        transitions=[
            ExpectedTransitionData(
                COMPLEX_SUSCEPTIBLE_NAME, COMPLEX_INFECTED_STATE_1_NAME, RateTransition, 0.2
            )
        ],
    )

    COMPLEX_INFECTED_1: ExpectedStateData = ExpectedStateData(
        name=COMPLEX_INFECTED_STATE_1_NAME,
        cause_type="sequela",
        allow_self_transition=False,
        prevalence=0.21,
        disability_weight=0.22,
        emr=0.23,
        transitions=[
            ExpectedTransitionData(
                COMPLEX_INFECTED_STATE_1_NAME,
                TRANSIENT_STATE_NAME,
                ProportionTransition,
                0.25,
            ),
            ExpectedTransitionData(
                COMPLEX_INFECTED_STATE_1_NAME,
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
                COMPLEX_STATE_3_NAME, COMPLEX_INFECTED_STATE_1_NAME, RateTransition, 0.95
            ),
            ExpectedTransitionData(
                COMPLEX_STATE_3_NAME, COMPLEX_SUSCEPTIBLE_NAME, RateTransition, 0.85
            ),
        ],
    )


STATES = ExpectedStates()


def complex_model_infected_1_prevalence(_, __):
    return STATES.COMPLEX_INFECTED_1.prevalence


def complex_model_3_to_1_transition_rate(_, __, ___):
    return STATES.COMPLEX_INFECTED_3.get_transitions()[COMPLEX_INFECTED_STATE_1_NAME].value


def complex_model_remission_rate(_, __, ___):
    return STATES.COMPLEX_INFECTED_3.get_transitions()[COMPLEX_SUSCEPTIBLE_NAME].value


class MockArtifactManager(MockArtifactManager_):
    def _load_artifact(self, _: str) -> MockArtifact:
        artifact = MockArtifact()

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
                "data_type": "rate",
                "data_sources": {
                    "incidence_rate": STATES.SIR_SUSCEPTIBLE.get_transitions()[
                        STATES.SIR_INFECTED.name
                    ].value
                },
            },
            "infected_remission": {
                "source": STATES.SIR_INFECTED.name,
                "sink": "recovered",
                "data_type": "rate",
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
        "states": {
            "susceptible": {},
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
                "data_sources": {
                    "birth_prevalence": STATES.COMPLEX_INFECTED_2.birth_prevalence,
                    "dwell_time": f"{STATES.COMPLEX_INFECTED_2.dwell_time} days",
                },
            },
            COMPLEX_STATE_3_NAME: {},
        },
        "transitions": {
            "infected_state_1_incidence": {
                "source": "susceptible",
                "sink": STATES.COMPLEX_INFECTED_1.name,
                "data_type": "rate",
                "data_sources": {
                    "incidence_rate": STATES.COMPLEX_SUSCEPTIBLE.get_transitions()[
                        STATES.COMPLEX_INFECTED_1.name
                    ].value
                },
            },
            "infected_state_1_to_transient": {
                "source": STATES.COMPLEX_INFECTED_1.name,
                "sink": TRANSIENT_STATE_NAME,
                "data_type": "proportion",
                "data_sources": {
                    "proportion": STATES.COMPLEX_INFECTED_1.get_transitions()[
                        TRANSIENT_STATE_NAME
                    ].value
                },
            },
            "infected_state_1_to_infected_state_2": {
                "source": STATES.COMPLEX_INFECTED_1.name,
                "sink": COMPLEX_STATE_2_NAME,
                "data_type": "proportion",
                "data_sources": {
                    "proportion": STATES.COMPLEX_INFECTED_1.get_transitions()[
                        COMPLEX_STATE_2_NAME
                    ].value
                },
            },
            "transient_to_infected_state_2": {
                "source": TRANSIENT_STATE_NAME,
                "sink": COMPLEX_STATE_2_NAME,
                "data_type": "proportion",
                "data_sources": {
                    "proportion": STATES.TRANSIENT.get_transitions()[
                        COMPLEX_STATE_2_NAME
                    ].value
                },
            },
            "infected_state_2_to_infected_state_3": {
                "source": COMPLEX_STATE_2_NAME,
                "sink": COMPLEX_STATE_3_NAME,
                "data_type": "dwell_time",
            },
            "infected_state_3_to_infected_state_1": {
                "source": COMPLEX_STATE_3_NAME,
                "sink": STATES.COMPLEX_INFECTED_1.name,
                "data_type": "rate",
                "data_sources": {
                    "transition_rate": "tests.plugins.test_parser::complex_model_3_to_1_transition_rate"
                },
            },
            "infected_state_3_remission": {
                "source": COMPLEX_STATE_3_NAME,
                "sink": "susceptible",
                "data_type": "rate",
                "data_sources": {
                    "transition_rate": "tests.plugins.test_parser::complex_model_remission_rate"
                },
            },
        },
    }
}


def create_simulation_config_tree(config_dict: Dict) -> ConfigTree:
    config_tree_layers = [
        "base",
        "user_configs",
        "component_configs",
        "model_override",
        "override",
    ]
    return ConfigTree(config_dict, layers=config_tree_layers)


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
def sim_components(base_config: ConfigTree, causes_config_parser_plugins: ConfigTree):
    simulation = InteractiveContext(
        components=create_simulation_config_tree(ALL_COMPONENTS_CONFIG_DICT),
        configuration=base_config,
        plugin_configuration=causes_config_parser_plugins,
    )
    return simulation.list_components()


def _test_parsing_of_config_file(
    component_config: ConfigTree,
    expected_component_names: Tuple[str] = (
        f"disease_model.{SIR_MODEL}",
        f"disease_model.{COMPLEX_MODEL}",
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
    "cause, expected_state_names",
    [
        (
            SIR_MODEL,
            {
                f"susceptible_state.{STATES.SIR_SUSCEPTIBLE.name}",
                f"disease_state.{STATES.SIR_INFECTED.name}",
                f"recovered_state.{STATES.SIR_RECOVERED.name}",
            },
        ),
        (
            COMPLEX_MODEL,
            {
                f"susceptible_state.{STATES.COMPLEX_SUSCEPTIBLE.name}",
                f"disease_state.{STATES.COMPLEX_INFECTED_1.name}",
                f"disease_state.{STATES.COMPLEX_INFECTED_2.name}",
                f"disease_state.{STATES.COMPLEX_INFECTED_3.name}",
                f"transient_disease_state.{STATES.TRANSIENT.name}",
            },
        ),
    ],
)
def test_disease_model(
    sim_components: Dict[str, Component], cause: str, expected_state_names: Set[str]
):
    model = sim_components[f"disease_model.{cause}"]
    assert isinstance(model, DiseaseModel)

    # the disease model's states have the expected names
    actual_state_names = {state.name for state in model.sub_components}
    assert actual_state_names == expected_state_names


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
        assert state.prevalence.data == expected_state_data.prevalence
        assert state.birth_prevalence.data == expected_state_data.birth_prevalence
        assert state.dwell_time.source.data == expected_state_data.dwell_time
        assert state.base_disability_weight.data == expected_state_data.disability_weight
        assert state.base_excess_mortality_rate.data == expected_state_data.emr

    # test that it has the expected transitions
    for transition in state.transition_set.transitions:
        expected_transition_data = expected_state_data.get_transitions()[
            transition.output_state.state_id
        ]
        assert type(transition) == expected_transition_data.transition_type
        if isinstance(transition, RateTransition):
            assert transition.base_rate.data == expected_transition_data.value
        elif isinstance(transition, ProportionTransition):
            assert transition.proportion.data == expected_transition_data.value


def test_invalid_data_source_throws_error():
    invalid_data_source_config_dict = {
        "causes": {
            "model_name": {
                "states": {
                    "susceptible": {},
                    "infected": {"data_sources": {"prevalence": "bad_data_source"}},
                },
                "transitions": {},
            }
        }
    }

    config = create_simulation_config_tree(invalid_data_source_config_dict)
    with pytest.raises(ValueError, match="Invalid data source"):
        CausesConfigurationParser().parse_component_config(config)
