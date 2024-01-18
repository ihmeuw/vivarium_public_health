from importlib import import_module
from typing import Any, Callable, Dict, List, Union

import pandas as pd
from loguru import logger
from pkg_resources import resource_filename
from vivarium import Component, ConfigTree
from vivarium.config_tree import ConfigurationError
from vivarium.framework.components import ComponentConfigurationParser
from vivarium.framework.engine import Builder
from vivarium.framework.state_machine import Trigger
from vivarium.framework.utilities import import_by_path

from vivarium_public_health.disease import (
    BaseDiseaseState,
    DiseaseModel,
    DiseaseState,
    RecoveredState,
    SusceptibleState,
    TransientDiseaseState,
)
from vivarium_public_health.utilities import TargetString


class CausesConfigurationParser(ComponentConfigurationParser):
    """
    Component configuration parser that acts the same as the standard vivarium
    `ComponentConfigurationParser` but adds the additional ability to parse a
    `causes` key and create `DiseaseModel` components.
    """

    def parse_component_config(self, component_config: ConfigTree) -> List[Component]:
        """
        Parses the component configuration and returns a list of components.

        In particular, this method looks for an `external_configuration` key and
        a `causes` key.

        The `external_configuration` key should have names of packages that
        contain cause model configuration files. Within that key should be a list
        of paths to cause model configuration files relative to the package.

        .. code-block:: yaml

            external_configuration:
                some_package:
                    - some/path/cause_model_1.yaml
                    - some/path/cause_model_2.yaml

        The `causes` key should contain configuration information for cause
        models.

        Note that this method modifies the simulation's component configuration
        by adding the contents of external configuration files to the
        `model_override` layer and adding default cause model configuration
        values for all cause models to the `component_config` layer.

        Parameters
        ----------
        component_config
            A ConfigTree defining the components to initialize.

        Returns
        -------
        List[Component]
            A list of initialized components.

        Raises
        ------
        ConfigurationError
            If the cause model configuration is invalid
        """
        components = []

        if "external_configuration" in component_config:
            self.validate_external_configuration(component_config["external_configuration"])
            for package, config_files in component_config["external_configuration"].items():
                for config_file in config_files.get_value():
                    source = f"{package}::{config_file}"
                    config_file = resource_filename(package, config_file)

                    external_config = ConfigTree(config_file)
                    component_config.update(
                        external_config, layer="model_override", source=source
                    )

        if "causes" in component_config:
            causes_config = component_config["causes"]
            self.validate_causes_config(causes_config)
            self.add_default_config_layer(causes_config)
            components += self.get_cause_model_components(causes_config)

        # Parse standard components (i.e. not cause models)
        standard_component_config = component_config.to_dict()
        standard_component_config.pop("external_configuration", None)
        standard_component_config.pop("causes", None)
        standard_components = (
            self.process_level(standard_component_config, [])
            if standard_component_config
            else []
        )

        return components + standard_components

    #########################
    # Configuration methods #
    #########################

    @staticmethod
    def add_default_config_layer(causes_config: ConfigTree) -> None:
        """
        Adds a default layer to the provided configuration that specifies
        default values for the cause model configuration.

        Parameters
        ----------
        causes_config
            A ConfigTree defining the cause model configurations

        Returns
        -------
        None
        """
        default_config = {}
        for cause_name, cause_config in causes_config.items():
            default_states_config = {}
            default_transitions_config = {}
            default_config[cause_name] = {
                "model_type": f"{DiseaseModel.__module__}.{DiseaseModel.__name__}",
                "initial_state": None,
                "states": default_states_config,
                "transitions": default_transitions_config,
            }

            for state_name, state_config in cause_config.states.items():
                default_states_config[state_name] = {
                    "cause_type": "cause",
                    "transient": False,
                    "allow_self_transition": True,
                    "side_effect": None,
                    "cleanup_function": None,
                    "state_type": None,
                }

            for transition_name, transition_config in cause_config.transitions.items():
                default_transitions_config[transition_name] = {"triggered": "NOT_TRIGGERED"}

        causes_config.update(
            default_config, layer="component_configs", source="causes_configuration_parser"
        )

    ################################
    # Cause model creation methods #
    ################################

    def get_cause_model_components(self, causes_config: ConfigTree) -> List[Component]:
        """
        Parses the cause model configuration and returns a list of
        `DiseaseModel` components.

        Parameters
        ----------
        causes_config
            A ConfigTree defining the cause model components to initialize

        Returns
        -------
        List[Component]
            A list of initialized `DiseaseModel` components
        """
        cause_models = []

        for cause_name, cause_config in causes_config.items():
            states: Dict[str, BaseDiseaseState] = {
                state_name: self.get_state(state_name, state_config, cause_name)
                for state_name, state_config in cause_config.states.items()
            }

            for transition_config in cause_config.transitions.values():
                self.add_transition(
                    states[transition_config.source],
                    states[transition_config.sink],
                    transition_config,
                )

            model_type = import_by_path(cause_config.model_type)
            initial_state = states.get(cause_config.initial_state, None)
            model = model_type(
                cause_name, initial_state=initial_state, states=list(states.values())
            )
            cause_models.append(model)

        return cause_models

    def get_state(
        self, state_name: str, state_config: ConfigTree, cause_name: str
    ) -> BaseDiseaseState:
        """
        Parses a state configuration and returns an initialized `BaseDiseaseState`
        object.

        Parameters
        ----------
        state_name
            The name of the state to initialize
        state_config
            A ConfigTree defining the state to initialize
        cause_name
            The name of the cause to which the state belongs

        Returns
        -------
        BaseDiseaseState
            An initialized `BaseDiseaseState` object
        """
        state_id = cause_name if state_name in ["susceptible", "recovered"] else state_name
        state_kwargs = {
            "cause_type": state_config.cause_type,
            "allow_self_transition": state_config.allow_self_transition,
        }
        if state_config.side_effect:
            # todo handle side effects properly
            state_kwargs["side_effect"] = lambda *x: x
        if state_config.cleanup_function:
            # todo handle cleanup functions properly
            state_kwargs["cleanup_function"] = lambda *x: x
        if "data_sources" in state_config:
            data_sources_config = state_config.data_sources
            state_kwargs["get_data_functions"] = {
                name: self.get_data_source(name, data_sources_config[name])
                for name in data_sources_config.keys()
            }

        if state_config.state_type is not None:
            state_type = import_by_path(state_config.state_type)
        elif state_config.transient:
            state_type = TransientDiseaseState
        elif state_name == "susceptible":
            state_type = SusceptibleState
        elif state_name == "recovered":
            state_type = RecoveredState
        else:
            state_type = DiseaseState

        state = state_type(state_id, **state_kwargs)
        return state

    def add_transition(
        self,
        source_state: BaseDiseaseState,
        sink_state: BaseDiseaseState,
        transition_config: ConfigTree,
    ) -> None:
        """
        Adds a transition between two states.

        Parameters
        ----------
        source_state
            The state the transition starts from
        sink_state
            The state the transition ends at
        transition_config
            A `ConfigTree` defining the transition to add

        Returns
        -------
        None
        """
        triggered = Trigger[transition_config.triggered]
        if "data_sources" in transition_config:
            data_sources_config = transition_config.data_sources
            data_sources = {
                name: self.get_data_source(name, data_sources_config[name])
                for name in data_sources_config.keys()
            }
        else:
            data_sources = None

        if transition_config.type == "rate":
            source_state.add_rate_transition(
                sink_state, get_data_functions=data_sources, triggered=triggered
            )
        elif transition_config.type == "proportion":
            source_state.add_proportion_transition(
                sink_state, get_data_functions=data_sources, triggered=triggered
            )
        elif transition_config.type == "dwell_time":
            source_state.add_dwell_time_transition(sink_state, triggered=triggered)
        else:
            raise ValueError(
                f"Invalid transition data type '{transition_config.type}'"
                f" provided for transition '{transition_config}'."
            )

    @staticmethod
    def get_data_source(
        name: str, source: Union[str, float]
    ) -> Callable[[Builder, Any], Any]:
        """
        Parses a data source and returns a callable that can be used to retrieve
        the data.

        Parameters
        ----------
        name
            The name of the data getter
        source
            The data source to parse

        Returns
        -------
        Callable[[Builder, Any], Any]
            A callable that can be used to retrieve the data
        """
        if isinstance(source, float):
            return lambda builder, *_: source

        try:
            timedelta = pd.Timedelta(source)
            return lambda builder, *_: timedelta
        except ValueError:
            pass

        if "::" in source:
            module, method = source.split("::")
            return getattr(import_module(module), method)

        try:
            target_string = TargetString(source)
            return lambda builder, *_: builder.data.load(target_string)
        except ValueError:
            pass

        raise ValueError(f"Invalid data source '{source}' for '{name}'.")

    ######################
    # Validation methods #
    ######################

    ALLOWABLE_CAUSE_KEYS = {"model_type", "initial_state", "states", "transitions"}
    ALLOWABLE_STATE_KEYS = {
        "state_type",
        "cause_type",
        "transient",
        "allow_self_transition",
        "side_effect",
        "data_sources",
        "cleanup_function",
    }

    ALLOWABLE_DATA_SOURCE_KEYS = {
        "state": {
            "prevalence",
            "birth_prevalence",
            "dwell_time",
            "disability_weight",
            "excess_mortality_rate",
        },
        "rate_transition": {
            "incidence_rate",
            "transition_rate",
            "remission_rate",
        },
        "proportion_transition": {"proportion"},
    }
    ALLOWABLE_TRANSITION_KEYS = {"source", "sink", "type", "triggered", "data_sources"}
    ALLOWABLE_TRANSITION_TYPE_KEYS = {"rate", "proportion", "dwell_time"}

    @staticmethod
    def validate_external_configuration(external_configuration: ConfigTree) -> None:
        """
        Validates the external configuration.

        Parameters
        ----------
        external_configuration
            A ConfigTree defining the external configuration

        Returns
        -------
        None

        Raises
        ------
        ConfigurationError
            If the external configuration is invalid
        """
        external_configuration = external_configuration.to_dict()
        error_messages = []
        for package, config_files in external_configuration.items():
            if not isinstance(package, str):
                error_messages.append(
                    f"Key '{package}' must be a string definition of a package."
                )
            if not isinstance(config_files, list):
                error_messages.append(
                    f"External configuration for package '{package}' must be a list."
                )
            else:
                for config_file in config_files:
                    if not isinstance(config_file, str) or not config_file.endswith(".yaml"):
                        error_messages.append(
                            f"External configuration for package '{package}' must "
                            "be a list of paths to yaml files."
                        )
        if error_messages:
            raise ConfigurationError("\n".join(error_messages), None)

    def validate_causes_config(self, causes_config: ConfigTree) -> None:
        """
        Validates the cause model configuration.

        Parameters
        ----------
        causes_config
            A ConfigTree defining the cause model configurations

        Returns
        -------
        None

        Raises
        ------
        ConfigurationError
            If the cause model configuration is invalid
        """
        causes_config = causes_config.to_dict()
        error_messages = []
        for cause_name, cause_config in causes_config.items():
            error_messages += self._validate_cause(cause_name, cause_config)

        if error_messages:
            raise ConfigurationError("\n".join(error_messages), None)

    def _validate_cause(self, cause_name: str, cause_config: Dict[str, Any]) -> List[str]:
        """
        Validates a cause configuration and returns a list of error messages.

        Parameters
        ----------
        cause_name
            The name of the cause to validate
        cause_config
            A ConfigTree defining the cause to validate

        Returns
        -------
        List[str]
            A list of error messages
        """
        error_messages = []
        if not isinstance(cause_config, dict):
            error_messages.append(
                f"Cause configuration for cause '{cause_name}' must be a dictionary."
            )
            return error_messages

        if not set(cause_config.keys()).issubset(
            CausesConfigurationParser.ALLOWABLE_CAUSE_KEYS
        ):
            error_messages.append(
                f"Cause configuration for cause '{cause_name}' may only"
                " contain the following keys: "
                f"{CausesConfigurationParser.ALLOWABLE_CAUSE_KEYS}."
            )

        model_type = cause_config.get("model_type", "")
        if model_type and not (isinstance(model_type, str) and "." in model_type):
            error_messages.append(
                f"If 'model_type' is provided for cause '{cause_name}' it "
                "must be a fully qualified import path for the type of the "
                f"desired model. Provided'{model_type}'."
            )

        states_config = cause_config.get("states", {})
        if not states_config:
            error_messages.append(
                f"Cause configuration for cause '{cause_name}' must define "
                "at least one state."
            )

        if not isinstance(states_config, dict):
            error_messages.append(
                f"States configuration for cause '{cause_name}' must be a dictionary."
            )
        else:
            initial_state = cause_config.get("initial_state", None)
            if initial_state is not None and initial_state not in states_config:
                error_messages.append(
                    f"Initial state '{cause_config['initial_state']}' for cause "
                    f"'{cause_name}' must be present in the states for cause "
                    f"'{cause_name}."
                )
            for state_name, state_config in states_config.items():
                error_messages += self._validate_state(cause_name, state_name, state_config)

        transitions_config = cause_config.get("transitions", {})
        can_parse_transitions = bool(transitions_config)
        if not isinstance(transitions_config, dict):
            can_parse_transitions = False
            error_messages.append(
                f"Transitions configuration for cause '{cause_name}' must be "
                "a dictionary if it is present."
            )

        if can_parse_transitions:
            for _, transition_config in cause_config["transitions"].items():
                error_messages += self._validate_transition(
                    cause_name, transition_config, states_config
                )

        return error_messages

    def _validate_state(
        self, cause_name: str, state_name: str, state_config: Dict[str, Any]
    ) -> List[str]:
        """
        Validates a state configuration and returns a list of error messages.

        Parameters
        ----------
        cause_name
            The name of the cause to which the state belongs
        state_name
            The name of the state to validate
        state_config
            A ConfigTree defining the state to validate

        Returns
        -------
        List[str]
            A list of error messages
        """
        error_messages = []

        if not isinstance(state_config, dict):
            error_messages.append(
                f"State configuration for in cause '{cause_name}' and "
                f"state '{state_name}' must be a dictionary."
            )
            return error_messages

        allowable_keys = set(CausesConfigurationParser.ALLOWABLE_STATE_KEYS)
        if state_name in ["susceptible", "recovered"]:
            allowable_keys.remove("data_sources")

        if not set(state_config.keys()).issubset(allowable_keys):
            error_messages.append(
                f"State configuration for in cause '{cause_name}' and "
                f"state '{state_name}' may only contain the following "
                f"keys: {allowable_keys}."
            )

        state_type = state_config.get("state_type", "")
        if state_type and not (isinstance(state_type, str) and "." in state_type):
            error_messages.append(
                f"If 'model_type' is provided for cause '{cause_name}' it "
                "must be a fully qualified import path for the type of the "
                f"desired model. Provided'{state_type}'."
            )

        if not isinstance(state_config.get("cause_type", ""), str):
            error_messages.append(
                f"Cause type for state '{state_name}' in cause '{cause_name}' "
                f"must be a string. Provided {state_config['cause_type']}."
            )
        is_transient = state_config.get("transient", False)
        if not isinstance(is_transient, bool):
            error_messages.append(
                f"Transient flag for state '{state_name}' in cause '{cause_name}' "
                f"must be a boolean. Provided {state_config['transient']}."
            )

        if state_type and is_transient:
            logger.warning(
                f"State '{state_name}' in cause '{cause_name}' has explicitly "
                f"configured type {state_type}, but has also provided a "
                "transient flag. This flag will be ignored."
            )

        if not isinstance(state_config.get("allow_self_transition", True), bool):
            error_messages.append(
                f"Allow self transition flag for state '{state_name}' in cause '{cause_name}' "
                f"must be a boolean. Provided {state_config['allow_self_transition']}."
            )

        if state_config.get("side_effect", None) is not None:
            logger.warning(
                f"Side effect for state '{state_name}' in cause '{cause_name}' "
                "is not supported and will be ignored."
            )

        if state_config.get("cleanup_function", None) is not None:
            logger.warning(
                f"Cleanup function for state '{state_name}' in cause '{cause_name}' "
                "is not supported and will be ignored."
            )

        error_messages += self._validate_data_sources(state_config, cause_name, "state")

        return error_messages

    def _validate_transition(
        self,
        cause_name: str,
        transition_config: Dict[str, Any],
        states_config: Dict[str, Any],
    ) -> List[str]:
        """
        Validates a transition configuration and returns a list of error messages.

        Parameters
        ----------
        cause_name
            The name of the cause to which the transition belongs
        transition_config
            A ConfigTree defining the transition to validate

        Returns
        -------
        List[str]
            A list of error messages
        """
        error_messages = []

        if not isinstance(transition_config, dict):
            error_messages.append(
                f"Transition configuration for in cause '{cause_name}' and "
                f"transition '{transition_config}' must be a dictionary."
            )
            return error_messages

        if not set(transition_config.keys()).issubset(
            CausesConfigurationParser.ALLOWABLE_TRANSITION_KEYS
        ):
            error_messages.append(
                f"Transition configuration for in cause '{cause_name}' and "
                f"transition '{transition_config}' may only contain the "
                f"following keys: {CausesConfigurationParser.ALLOWABLE_TRANSITION_KEYS}."
            )
        source = transition_config.get("source", None)
        sink = transition_config.get("sink", None)
        if sink is None or source is None:
            error_messages.append(
                f"Transition configuration for in cause '{cause_name}' and "
                f"transition '{transition_config}' must contain both a source "
                f"and a sink."
            )

        if source not in states_config:
            error_messages.append(
                f"Transition configuration for in cause '{cause_name}' and "
                f"transition '{transition_config}' must contain a source that "
                f"is present in the states."
            )
        if sink not in states_config:
            error_messages.append(
                f"Transition configuration for in cause '{cause_name}' and "
                f"transition '{transition_config}' must contain a sink that "
                f"is present in the states."
            )
        transition_type = transition_config.get("type", "rate")
        if transition_type not in CausesConfigurationParser.ALLOWABLE_TRANSITION_TYPE_KEYS:
            error_messages.append(
                f"Transition configuration for in cause '{cause_name}' and "
                f"transition '{transition_config}' may only have one of the following "
                f"values: {CausesConfigurationParser.ALLOWABLE_TRANSITION_TYPE_KEYS}."
            )
        if (
            "triggered" in transition_config
            and transition_config["triggered"] not in Trigger.__members__
        ):
            error_messages.append(
                f"Transition configuration for in cause '{cause_name}' and "
                f"transition '{transition_config}' may only have one of the following "
                f"values: {Trigger.__members__}."
            )
        if transition_type == "dwell_time" and "data_sources" in transition_config:
            error_messages.append(
                f"Transition configuration for in cause '{cause_name}' and "
                f"transition '{transition_config}' is a dwell-time transition and "
                f"may not have data sources as dwell-time is configured on the state."
            )
        else:
            error_messages += self._validate_data_sources(
                transition_config, cause_name, f"{transition_type}_transition"
            )
        return error_messages

    def _validate_data_sources(
        self, config: Dict[str, Any], cause_name: str, config_type: str
    ) -> List[str]:
        """
        Validates the data sources in a configuration and returns a list of
        error messages.

        Parameters
        ----------
        config
            A ConfigTree defining the configuration to validate
        cause_name
            The name of the cause to which the configuration belongs
        config_type
            The type of the configuration to validate

        Returns
        -------
        List[str]
            A list of error messages
        """
        error_messages = []
        data_sources_config = config.get("data_sources", None)
        if data_sources_config is not None:
            if not isinstance(data_sources_config, dict):
                error_messages.append(
                    f"Data sources configuration for {config_type} '{config}' in "
                    f"cause '{cause_name}' must be a dictionary if it is present."
                )
            else:
                if not set(data_sources_config.keys()).issubset(
                    CausesConfigurationParser.ALLOWABLE_DATA_SOURCE_KEYS[config_type]
                ):
                    error_messages.append(
                        f"Data sources configuration for {config_type} '{config}' "
                        f"in cause '{cause_name}' may only contain the following keys: "
                        f"{CausesConfigurationParser.ALLOWABLE_DATA_SOURCE_KEYS[config_type]}."
                    )

                for name, source in data_sources_config.items():
                    try:
                        self.get_data_source(name, source)
                    except ValueError:
                        error_messages.append(
                            f"Configuration for {config_type} '{config}' in "
                            f"cause '{cause_name}' has an invalid data source at "
                            f"'{source}'."
                        )
        return error_messages
