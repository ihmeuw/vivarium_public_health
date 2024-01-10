import dataclasses
from typing import Dict, List, Set, Type

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
class TransitionData:
    source: str
    sink: str
    transition_type: Type[Transition]
    value: float


@dataclasses.dataclass
class StateData:
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

    transitions: List[TransitionData] = dataclasses.field(default_factory=list)

    def get_transitions(self) -> Dict[str, TransitionData]:
        """Return a dict of transitions keyed by their sink state name."""
        return {transition.sink: transition for transition in self.transitions}


SIR_SUSCEPTIBLE_STATE = StateData(
    name=SIR_SUSCEPTIBLE_NAME,
    state_type=SusceptibleState,
    transitions=[
        TransitionData(SIR_SUSCEPTIBLE_NAME, SIR_INFECTED_STATE_NAME, RateTransition, 0.5)
    ],
)
SIR_INFECTED = StateData(
    name=SIR_INFECTED_STATE_NAME,
    prevalence=0.11,
    disability_weight=0.12,
    emr=0.13,
    transitions=[
        TransitionData(SIR_INFECTED_STATE_NAME, SIR_RECOVERED_NAME, RateTransition, 0.6)
    ],
)
SIR_RECOVERED_STATE = StateData(name=SIR_RECOVERED_NAME, state_type=RecoveredState)

COMPLEX_SUSCEPTIBLE_STATE = StateData(
    name=COMPLEX_SUSCEPTIBLE_NAME,
    state_type=SusceptibleState,
    transitions=[
        TransitionData(
            COMPLEX_SUSCEPTIBLE_NAME, COMPLEX_INFECTED_STATE_1_NAME, RateTransition, 0.2
        )
    ],
)

COMPLEX_INFECTED_STATE_1 = StateData(
    name=COMPLEX_INFECTED_STATE_1_NAME,
    cause_type="sequela",
    allow_self_transition=False,
    prevalence=0.21,
    disability_weight=0.22,
    emr=0.23,
    transitions=[
        TransitionData(
            COMPLEX_INFECTED_STATE_1_NAME, TRANSIENT_STATE_NAME, ProportionTransition, 0.25
        ),
        TransitionData(
            COMPLEX_INFECTED_STATE_1_NAME, COMPLEX_STATE_2_NAME, ProportionTransition, 0.75
        ),
    ],
)
TRANSIENT_STATE = StateData(
    name=TRANSIENT_STATE_NAME,
    state_type=TransientDiseaseState,
    is_transient=True,
    transitions=[
        TransitionData(TRANSIENT_STATE_NAME, COMPLEX_STATE_2_NAME, ProportionTransition, 1.0),
    ],
)
COMPLEX_INFECTED_STATE_2 = StateData(
    name=COMPLEX_STATE_2_NAME,
    prevalence=0.31,
    birth_prevalence=0.3,
    dwell_time=3.0,
    disability_weight=0.32,
    emr=0.33,
    transitions=[
        TransitionData(COMPLEX_STATE_2_NAME, COMPLEX_STATE_3_NAME, Transition, 0.0),
    ],
)

COMPLEX_INFECTED_STATE_3 = StateData(
    name=COMPLEX_STATE_3_NAME,
    prevalence=0.41,
    disability_weight=0.42,
    emr=0.43,
    transitions=[
        TransitionData(
            COMPLEX_STATE_3_NAME, COMPLEX_INFECTED_STATE_1_NAME, RateTransition, 0.95
        ),
        TransitionData(COMPLEX_STATE_3_NAME, COMPLEX_SUSCEPTIBLE_NAME, RateTransition, 0.85),
    ],
)

STATES_TO_TEST = [
    SIR_SUSCEPTIBLE_STATE,
    SIR_INFECTED,
    SIR_RECOVERED_STATE,
    COMPLEX_SUSCEPTIBLE_STATE,
    COMPLEX_INFECTED_STATE_1,
    TRANSIENT_STATE,
    COMPLEX_INFECTED_STATE_2,
    COMPLEX_INFECTED_STATE_3,
]


def complex_model_infected_1_prevalence(_, __):
    return COMPLEX_INFECTED_STATE_1.prevalence


def complex_model_3_to_1_transition_rate(_, __, ___):
    return COMPLEX_INFECTED_STATE_3.get_transitions()[COMPLEX_INFECTED_STATE_1_NAME].value


def complex_model_remission_rate(_, __, ___):
    return COMPLEX_INFECTED_STATE_3.get_transitions()[COMPLEX_SUSCEPTIBLE_NAME].value


class MockArtifactManager(MockArtifactManager_):
    def _load_artifact(self, _: str) -> MockArtifact:
        artifact = MockArtifact()

        artifact.mocks[f"cause.{SIR_INFECTED.name}.prevalence"] = SIR_INFECTED.prevalence
        artifact.mocks[
            f"cause.{SIR_INFECTED.name}.disability_weight"
        ] = SIR_INFECTED.disability_weight
        artifact.mocks[
            f"cause.{SIR_INFECTED.name}.excess_mortality_rate"
        ] = SIR_INFECTED.emr

        artifact.mocks[
            "cause.some_custom_cause.disability_weight"
        ] = COMPLEX_INFECTED_STATE_1.disability_weight
        artifact.mocks[
            "cause.some_custom_cause.excess_mortality_rate"
        ] = COMPLEX_INFECTED_STATE_1.emr

        artifact.mocks[
            f"cause.{COMPLEX_STATE_2_NAME}.prevalence"
        ] = COMPLEX_INFECTED_STATE_2.prevalence
        artifact.mocks[
            f"cause.{COMPLEX_STATE_2_NAME}.disability_weight"
        ] = COMPLEX_INFECTED_STATE_2.disability_weight
        artifact.mocks[
            f"cause.{COMPLEX_STATE_2_NAME}.excess_mortality_rate"
        ] = COMPLEX_INFECTED_STATE_2.emr

        artifact.mocks[
            f"cause.{COMPLEX_STATE_3_NAME}.prevalence"
        ] = COMPLEX_INFECTED_STATE_3.prevalence
        artifact.mocks[
            f"cause.{COMPLEX_STATE_3_NAME}.disability_weight"
        ] = COMPLEX_INFECTED_STATE_3.disability_weight
        artifact.mocks[
            f"cause.{COMPLEX_STATE_3_NAME}.excess_mortality_rate"
        ] = COMPLEX_INFECTED_STATE_3.emr

        return artifact


def get_component_config() -> ConfigTree:
    component_config = {
        "causes": {
            SIR_MODEL: {
                "states": {
                    "susceptible": {},
                    SIR_INFECTED.name: {},
                    "recovered": {},
                },
                "transitions": {
                    "infected_incidence": {
                        "source": "susceptible",
                        "sink": SIR_INFECTED.name,
                        "data_type": "rate",
                        "data_sources": {
                            "incidence_rate": SIR_SUSCEPTIBLE_STATE.get_transitions()[
                                SIR_INFECTED.name
                            ].value
                        },
                    },
                    "infected_remission": {
                        "source": SIR_INFECTED.name,
                        "sink": "recovered",
                        "data_type": "rate",
                        "data_sources": {
                            "remission_rate": SIR_INFECTED.get_transitions()[
                                f"recovered_from_{SIR_MODEL}"
                            ].value
                        },
                    },
                },
            },
            COMPLEX_MODEL: {
                "states": {
                    "susceptible": {},
                    COMPLEX_INFECTED_STATE_1.name: {
                        "cause_type": COMPLEX_INFECTED_STATE_1.cause_type,
                        "allow_self_transition": COMPLEX_INFECTED_STATE_1.allow_self_transition,
                        "data_sources": {
                            "prevalence": "tests.plugins.test_parser::complex_model_infected_1_prevalence",
                            "disability_weight": "cause.some_custom_cause.disability_weight",
                            "excess_mortality_rate": "cause.some_custom_cause.excess_mortality_rate",
                        },
                    },
                    TRANSIENT_STATE_NAME: {"transient": True},
                    COMPLEX_STATE_2_NAME: {
                        "data_sources": {
                            "birth_prevalence": COMPLEX_INFECTED_STATE_2.birth_prevalence,
                            "dwell_time": f"{COMPLEX_INFECTED_STATE_2.dwell_time} days",
                        },
                    },
                    COMPLEX_STATE_3_NAME: {},
                },
                "transitions": {
                    "infected_state_1_incidence": {
                        "source": "susceptible",
                        "sink": COMPLEX_INFECTED_STATE_1.name,
                        "data_type": "rate",
                        "data_sources": {
                            "incidence_rate": COMPLEX_SUSCEPTIBLE_STATE.get_transitions()[
                                COMPLEX_INFECTED_STATE_1.name
                            ].value
                        },
                    },
                    "infected_state_1_to_transient": {
                        "source": COMPLEX_INFECTED_STATE_1.name,
                        "sink": TRANSIENT_STATE_NAME,
                        "data_type": "proportion",
                        "data_sources": {
                            "proportion": COMPLEX_INFECTED_STATE_1.get_transitions()[
                                TRANSIENT_STATE_NAME
                            ].value
                        },
                    },
                    "infected_state_1_to_infected_state_2": {
                        "source": COMPLEX_INFECTED_STATE_1.name,
                        "sink": COMPLEX_STATE_2_NAME,
                        "data_type": "proportion",
                        "data_sources": {
                            "proportion": COMPLEX_INFECTED_STATE_1.get_transitions()[
                                COMPLEX_STATE_2_NAME
                            ].value
                        },
                    },
                    "transient_to_infected_state_2": {
                        "source": TRANSIENT_STATE_NAME,
                        "sink": COMPLEX_STATE_2_NAME,
                        "data_type": "proportion",
                        "data_sources": {
                            "proportion": TRANSIENT_STATE.get_transitions()[
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
                        "sink": COMPLEX_INFECTED_STATE_1.name,
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


def test_parser_returns_list_of_components(component_list_parsed_from_config):
    assert isinstance(component_list_parsed_from_config, list)
    expected_component_names = {
        f"disease_model.{SIR_MODEL}",
        f"disease_model.{COMPLEX_MODEL}",
        "test_population",
    }
    assert expected_component_names == {
        component.name for component in component_list_parsed_from_config
    }


@pytest.mark.parametrize(
    "cause, expected_state_names",
    [
        (
            SIR_MODEL,
            {
                f"susceptible_state.susceptible_to_{SIR_MODEL}",
                f"disease_state.{SIR_INFECTED.name}",
                f"recovered_state.recovered_from_{SIR_MODEL}",
            },
        ),
        (
            COMPLEX_MODEL,
            {
                f"susceptible_state.susceptible_to_{COMPLEX_MODEL}",
                f"disease_state.{COMPLEX_INFECTED_STATE_1.name}",
                f"disease_state.{COMPLEX_STATE_2_NAME}",
                f"disease_state.{COMPLEX_STATE_3_NAME}",
                f"transient_disease_state.{TRANSIENT_STATE_NAME}",
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
    expected_state_names = {state.name for state in STATES_TO_TEST}
    assert actual_state_names == expected_state_names


@pytest.mark.parametrize(
    "expected_state_data", STATES_TO_TEST, ids=[state.name for state in STATES_TO_TEST]
)
def test_disease_state(sim_components: Dict[str, Component], expected_state_data: StateData):
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


# todo test config file references external config file
# todo test config file defines causes itself
# todo test config input as dict
# todo test config file with both local and external definition of causes


# todo test invalid data source
