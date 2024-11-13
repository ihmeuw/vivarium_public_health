"""
===============================
Component Configuration Parsers
===============================

Component Configuration Parsers in this module are specialized implementations of
:class:`ComponentConfigurationParser <vivarium.framework.components.parser.ComponentConfigurationParser>`
that can parse configurations of components specific to the Vivarium Public
Health package.

"""
import warnings
from collections.abc import Callable
from importlib import import_module
from typing import Any

import pandas as pd
from layered_config_tree import LayeredConfigTree
from pkg_resources import resource_filename
from vivarium import Component
from vivarium.framework.components import ComponentConfigurationParser
from vivarium.framework.components.parser import ParsingError
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


class CausesParsingErrors(ParsingError):
    """Error raised when there are any errors parsing a cause model configuration."""

    def __init__(self, messages: list[str]):
        super().__init__("\n - " + "\n - ".join(messages))


class CausesConfigurationParser(ComponentConfigurationParser):
    """Parser for cause model configurations.

    Component configuration parser that acts the same as the standard vivarium
    `ComponentConfigurationParser` but adds the additional ability to parse a
    configuration to create `DiseaseModel` components. These DiseaseModel
    configurations can either be specified directly in the configuration in a
    `causes` key or in external configuration files that are specified in the
    `external_configuration` key.

    """

    DEFAULT_MODEL_CONFIG = {
        "model_type": f"{DiseaseModel.__module__}.{DiseaseModel.__name__}",
        "initial_state": None,
        "residual_state": None,
    }
    """Default cause model configuration if it's not explicitly specified.
    
    Initial state and residual state cannot both be provided. If neither initial
    state nor residual state has been specified, the cause  model must have a
    state named 'susceptible'.
    """

    DEFAULT_STATE_CONFIG = {
        "cause_type": "cause",
        "transient": False,
        "allow_self_transition": True,
        "side_effect": None,
        "cleanup_function": None,
        "state_type": None,
    }
    """Default state configuration if it's not explicitly specified."""

    DEFAULT_TRANSITION_CONFIG = {"triggered": "NOT_TRIGGERED"}
    """Default triggered value.
    
    This value is used if the transition configuration does not explicity specify it.
    """

    def parse_component_config(self, component_config: LayeredConfigTree) -> list[Component]:
        """Parses the component configuration and returns a list of components.

        In particular, this method looks for an `external_configuration` key
        and/or a `causes` key.

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

        .. code-block:: yaml

            causes:
                cause_1:
                    model_type: vivarium_public_health.disease.DiseaseModel
                    residual_state: susceptible
                    states:
                        susceptible:
                            cause_type: cause
                            data_sources: {}
                        infected:
                            cause_type: cause
                            transient: false
                            allow_self_transition: true
                            data_sources: {}
                    transitions:
                        transition_1:
                            source: susceptible
                            sink: infected
                            transition_type: rate
                            data_sources: {}

        # todo add information about the data_sources configuration

        Note that this method modifies the simulation's component configuration
        by adding the contents of external configuration files to the
        `model_override` layer and adding default cause model configuration
        values for all cause models to the `component_config` layer.

        Parameters
        ----------
        component_config
            A LayeredConfigTree defining the components to initialize.

        Returns
        -------
            A list of initialized components.

        Raises
        ------
        CausesParsingErrors
            If the cause model configuration is invalid
        """
        components = []

        if "external_configuration" in component_config:
            self._validate_external_configuration(component_config["external_configuration"])
            for package, config_files in component_config["external_configuration"].items():
                for config_file in config_files.get_value():
                    source = f"{package}::{config_file}"
                    config_file = resource_filename(package, config_file)

                    external_config = LayeredConfigTree(config_file)
                    component_config.update(
                        external_config, layer="model_override", source=source
                    )

        if "causes" in component_config:
            causes_config = component_config["causes"]
            self._validate_causes_config(causes_config)
            self._add_default_config_layer(causes_config)
            components += self._get_cause_model_components(causes_config)

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

    def _add_default_config_layer(self, causes_config: LayeredConfigTree) -> None:
        """Adds a default layer to the provided configuration.

        This default layer specifies values for the cause model configuration.

        Parameters
        ----------
        causes_config
            A LayeredConfigTree defining the cause model configurations
        """
        default_config = {}
        for cause_name, cause_config in causes_config.items():
            default_states_config = {}
            default_transitions_config = {}
            default_config[cause_name] = {
                **self.DEFAULT_MODEL_CONFIG,
                "states": default_states_config,
                "transitions": default_transitions_config,
            }

            for state_name, state_config in cause_config.states.items():
                default_states_config[state_name] = self.DEFAULT_STATE_CONFIG

            for transition_name, transition_config in cause_config.transitions.items():
                default_transitions_config[transition_name] = self.DEFAULT_TRANSITION_CONFIG

        causes_config.update(
            default_config, layer="component_configs", source="causes_configuration_parser"
        )

    ################################
    # Cause model creation methods #
    ################################

    def _get_cause_model_components(
        self, causes_config: LayeredConfigTree
    ) -> list[Component]:
        """Parses the cause model configuration and returns the `DiseaseModel` components.

        Parameters
        ----------
        causes_config
            A LayeredConfigTree defining the cause model components to initialize

        Returns
        -------
            A list of initialized `DiseaseModel` components
        """
        cause_models = []

        for cause_name, cause_config in causes_config.items():
            data_sources = None
            if "data_sources" in cause_config:
                data_sources_config = cause_config.data_sources
                data_sources = self._get_data_sources(data_sources_config)

            states: dict[str, BaseDiseaseState] = {
                state_name: self._get_state(state_name, state_config, cause_name)
                for state_name, state_config in cause_config.states.items()
            }

            for transition_config in cause_config.transitions.values():
                self._add_transition(
                    states[transition_config.source],
                    states[transition_config.sink],
                    transition_config,
                )

            model_type = import_by_path(cause_config.model_type)
            residual_state = states.get(
                cause_config.residual_state, states.get(cause_config.initial_state, None)
            )
            model = model_type(
                cause_name,
                states=list(states.values()),
                residual_state=residual_state,
                get_data_functions=data_sources,
            )
            cause_models.append(model)

        return cause_models

    def _get_state(
        self, state_name: str, state_config: LayeredConfigTree, cause_name: str
    ) -> BaseDiseaseState:
        """Parses a state configuration and returns an initialized `BaseDiseaseState` object.

        Parameters
        ----------
        state_name
            The name of the state to initialize
        state_config
            A LayeredConfigTree defining the state to initialize
        cause_name
            The name of the cause to which the state belongs

        Returns
        -------
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
            state_kwargs["get_data_functions"] = self._get_data_sources(data_sources_config)

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

    def _add_transition(
        self,
        source_state: BaseDiseaseState,
        sink_state: BaseDiseaseState,
        transition_config: LayeredConfigTree,
    ) -> None:
        """Adds a transition between two states.

        Parameters
        ----------
        source_state
            The state the transition starts from
        sink_state
            The state the transition ends at
        transition_config
            A `LayeredConfigTree` defining the transition to add
        """
        triggered = Trigger[transition_config.triggered]
        if "data_sources" in transition_config:
            data_sources_config = transition_config.data_sources
            data_sources = self._get_data_sources(data_sources_config)
        else:
            data_sources = None

        if transition_config["transition_type"] == "rate":
            source_state.add_rate_transition(
                sink_state, get_data_functions=data_sources, triggered=triggered
            )
        elif transition_config["transition_type"] == "proportion":
            source_state.add_proportion_transition(
                sink_state, get_data_functions=data_sources, triggered=triggered
            )
        elif transition_config["transition_type"] == "dwell_time":
            source_state.add_dwell_time_transition(sink_state, triggered=triggered)
        else:
            raise ValueError(
                f"Invalid transition data type '{transition_config.type}'"
                f" provided for transition '{transition_config}'."
            )

    def _get_data_sources(
        self, config: LayeredConfigTree
    ) -> dict[str, Callable[[Builder, Any], Any]]:
        """Parses a data sources configuration and returns the data sources.

        Parameters
        ----------
        config
            A LayeredConfigTree defining the data sources to initialize

        Returns
        -------
            A dictionary of data source getters
        """
        return {name: self._get_data_source(name, config[name]) for name in config.keys()}

    @staticmethod
    def _get_data_source(name: str, source: str | float) -> Callable[[Builder, Any], Any]:
        """Parses a data source and returns a callable that can be used to retrieve the data.

        Parameters
        ----------
        name
            The name of the data getter
        source
            The data source to parse

        Returns
        -------
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

    _CAUSE_KEYS = {
        "model_type",
        "initial_state",
        "states",
        "transitions",
        "data_sources",
        "residual_state",
    }
    _STATE_KEYS = {
        "state_type",
        "cause_type",
        "transient",
        "allow_self_transition",
        "side_effect",
        "data_sources",
        "cleanup_function",
    }

    _DATA_SOURCE_KEYS = {
        "cause": {"cause_specific_mortality_rate"},
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
    _TRANSITION_KEYS = {"source", "sink", "transition_type", "triggered", "data_sources"}
    _TRANSITION_TYPE_KEYS = {"rate", "proportion", "dwell_time"}

    @staticmethod
    def _validate_external_configuration(external_configuration: LayeredConfigTree) -> None:
        """Validates the external configuration.

        Parameters
        ----------
        external_configuration
            A LayeredConfigTree defining the external configuration

        Raises
        ------
        CausesParsingErrors
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
            raise CausesParsingErrors(error_messages)

    def _validate_causes_config(self, causes_config: LayeredConfigTree) -> None:
        """Validates the cause model configuration.

        Parameters
        ----------
        causes_config
            A LayeredConfigTree defining the cause model configurations

        Raises
        ------
        CausesParsingErrors
            If the cause model configuration is invalid
        """
        causes_config = causes_config.to_dict()
        error_messages = []
        for cause_name, cause_config in causes_config.items():
            error_messages += self._validate_cause(cause_name, cause_config)

        if error_messages:
            raise CausesParsingErrors(error_messages)

    def _validate_cause(self, cause_name: str, cause_config: dict[str, Any]) -> list[str]:
        """Validates a cause configuration and returns a list of error messages.

        Parameters
        ----------
        cause_name
            The name of the cause to validate
        cause_config
            A LayeredConfigTree defining the cause to validate

        Returns
        -------
            A list of error messages
        """
        error_messages = []
        if not isinstance(cause_config, dict):
            error_messages.append(
                f"Cause configuration for cause '{cause_name}' must be a dictionary."
            )
            return error_messages

        if not set(cause_config.keys()).issubset(self._CAUSE_KEYS):
            error_messages.append(
                f"Cause configuration for cause '{cause_name}' may only"
                " contain the following keys: "
                f"{self._CAUSE_KEYS}."
            )

        if "model_type" in cause_config:
            error_messages += self._validate_imported_type(
                cause_config["model_type"], cause_name, "model"
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
            residual_state = cause_config.get("residual_state", None)
            if initial_state is not None:
                warnings.warn(
                    "In the future, the 'initial_state' cause configuration will"
                    " be used to initialize all simulants into that state. To"
                    " retain the current behavior of defining a residual state,"
                    " use the 'residual_state' cause configuration.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                if residual_state is None:
                    residual_state = initial_state
                else:
                    error_messages.append(
                        "A cause may not have both 'initial_state and"
                        " 'residual_state' configurations."
                    )

            if residual_state is not None and residual_state not in states_config:
                error_messages.append(
                    f"Residual state '{residual_state}' for cause '{cause_name}'"
                    f" must be present in the states for cause '{cause_name}."
                )
            for state_name, state_config in states_config.items():
                error_messages += self._validate_state(cause_name, state_name, state_config)

        transitions_config = cause_config.get("transitions", {})
        if not isinstance(transitions_config, dict):
            error_messages.append(
                f"Transitions configuration for cause '{cause_name}' must be "
                "a dictionary if it is present."
            )
        else:
            for transition_name, transition_config in transitions_config.items():
                error_messages += self._validate_transition(
                    cause_name, transition_name, transition_config, states_config
                )

        error_messages += self._validate_data_sources(
            cause_config, cause_name, "cause", cause_name
        )

        return error_messages

    def _validate_state(
        self, cause_name: str, state_name: str, state_config: dict[str, Any]
    ) -> list[str]:
        """Validates a state configuration and returns a list of error messages.

        Parameters
        ----------
        cause_name
            The name of the cause to which the state belongs
        state_name
            The name of the state to validate
        state_config
            A LayeredConfigTree defining the state to validate

        Returns
        -------
            A list of error messages
        """
        error_messages = []

        if not isinstance(state_config, dict):
            error_messages.append(
                f"State configuration for in cause '{cause_name}' and "
                f"state '{state_name}' must be a dictionary."
            )
            return error_messages

        allowable_keys = set(self._STATE_KEYS)
        if state_name in ["susceptible", "recovered"]:
            allowable_keys.remove("data_sources")

        if not set(state_config.keys()).issubset(allowable_keys):
            error_messages.append(
                f"State configuration for in cause '{cause_name}' and "
                f"state '{state_name}' may only contain the following "
                f"keys: {allowable_keys}."
            )

        state_type = state_config.get("state_type", "")
        error_messages += self._validate_imported_type(
            state_type, cause_name, "state", state_name
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

        if state_name in ["susceptible", "recovered"] and state_type:
            error_messages.append(
                f"The name '{state_name}' in cause '{cause_name}' concretely "
                f"specifies the state type, so state_type is not an allowed "
                "configuration."
            )

        if state_name in ["susceptible", "recovered"] and is_transient:
            error_messages.append(
                f"The name '{state_name}' in cause '{cause_name}' concretely "
                f"specifies the state type, so transient is not an allowed "
                "configuration."
            )

        if is_transient and state_type:
            error_messages.append(
                f"Specifying transient as True for state '{state_name}' in cause "
                f"'{cause_name}' concretely specifies the state type, so "
                "state_type is not an allowed configuration."
            )

        if not isinstance(state_config.get("allow_self_transition", True), bool):
            error_messages.append(
                f"Allow self transition flag for state '{state_name}' in cause "
                f"'{cause_name}' must be a boolean. Provided "
                f"'{state_config['allow_self_transition']}'."
            )

        error_messages += self._validate_data_sources(
            state_config, cause_name, "state", state_name
        )

        return error_messages

    def _validate_transition(
        self,
        cause_name: str,
        transition_name: str,
        transition_config: dict[str, Any],
        states_config: dict[str, Any],
    ) -> list[str]:
        """Validates a transition configuration and returns a list of error messages.

        Parameters
        ----------
        cause_name
            The name of the cause to which the transition belongs
        transition_name
            The name of the transition to validate
        transition_config
            A LayeredConfigTree defining the transition to validate
        states_config
            A LayeredConfigTree defining the states for the cause

        Returns
        -------
            A list of error messages
        """
        error_messages = []

        if not isinstance(transition_config, dict):
            error_messages.append(
                f"Transition configuration for in cause '{cause_name}' and "
                f"transition '{transition_name}' must be a dictionary."
            )
            return error_messages

        if not set(transition_config.keys()).issubset(
            CausesConfigurationParser._TRANSITION_KEYS
        ):
            error_messages.append(
                f"Transition configuration for in cause '{cause_name}' and "
                f"transition '{transition_name}' may only contain the "
                f"following keys: {self._TRANSITION_KEYS}."
            )
        source = transition_config.get("source", None)
        sink = transition_config.get("sink", None)
        if sink is None or source is None:
            error_messages.append(
                f"Transition configuration for in cause '{cause_name}' and "
                f"transition '{transition_name}' must contain both a source "
                f"and a sink."
            )

        if source is not None and source not in states_config:
            error_messages.append(
                f"Transition configuration for in cause '{cause_name}' and "
                f"transition '{transition_name}' must contain a source that "
                f"is present in the states."
            )

        if sink is not None and sink not in states_config:
            error_messages.append(
                f"Transition configuration for in cause '{cause_name}' and "
                f"transition '{transition_name}' must contain a sink that "
                f"is present in the states."
            )

        if (
            "triggered" in transition_config
            and transition_config["triggered"] not in Trigger.__members__
        ):
            error_messages.append(
                f"Transition configuration for in cause '{cause_name}' and "
                f"transition '{transition_name}' may only have one of the following "
                f"values: {Trigger.__members__}."
            )

        if "transition_type" not in transition_config:
            error_messages.append(
                f"Transition configuration for in cause '{cause_name}' and "
                f"transition '{transition_name}' must contain a transition type."
            )
        else:
            transition_type = transition_config["transition_type"]
            if transition_type not in self._TRANSITION_TYPE_KEYS:
                error_messages.append(
                    f"Transition configuration for in cause '{cause_name}' and "
                    f"transition '{transition_name}' may only contain the "
                    f"following values: {self._TRANSITION_TYPE_KEYS}."
                )
            if transition_type == "dwell_time" and "data_sources" in transition_config:
                error_messages.append(
                    f"Transition configuration for in cause '{cause_name}' and "
                    f"transition '{transition_name}' is a dwell-time transition and "
                    f"may not have data sources as dwell-time is configured on the state."
                )
            elif transition_type in self._TRANSITION_TYPE_KEYS.difference({"dwell_time"}):
                error_messages += self._validate_data_sources(
                    transition_config,
                    cause_name,
                    f"{transition_type}_transition",
                    transition_name,
                )
        return error_messages

    @staticmethod
    def _validate_imported_type(
        import_path: str, cause_name: str, entity_type: str, entity_name: str | None = None
    ) -> list[str]:
        """Validates an imported type and returns a list of error messages.

        Parameters
        ----------
        import_path
            The import path to validate
        cause_name
            The name of the cause to which the imported type belongs
        entity_type
            The type of the entity to which the imported type belongs
        entity_name
            The name of the entity to which the imported type belongs, if it is
            not a cause

        Returns
        -------
            A list of error messages
        """
        expected_type = {"model": DiseaseModel, "state": BaseDiseaseState}[entity_type]

        error_messages = []
        if not import_path:
            return error_messages

        try:
            imported_type = import_by_path(import_path)
            if not (
                isinstance(imported_type, type) and issubclass(imported_type, expected_type)
            ):
                raise TypeError
        except (ModuleNotFoundError, AttributeError, TypeError, ValueError):
            error_messages.append(
                f"If '{entity_type}_type' is provided for cause '{cause_name}' "
                f"{f'and {entity_type} {entity_name} ' if entity_name else ''}it "
                f"must be the fully qualified import path to a {expected_type} "
                f"implementation. Provided'{import_path}'."
            )
        return error_messages

    def _validate_data_sources(
        self, config: dict[str, Any], cause_name: str, config_type: str, config_name: str
    ) -> list[str]:
        """Validates the data sources in a configuration and returns any error messages.

        Parameters
        ----------
        config
            A LayeredConfigTree defining the configuration to validate
        cause_name
            The name of the cause to which the configuration belongs
        config_type
            The type of the configuration to validate
        config_name
            The name of the configuration being validated

        Returns
        -------
            A list of error messages
        """
        error_messages = []
        data_sources_config = config.get("data_sources", {})
        if not isinstance(data_sources_config, dict):
            error_messages.append(
                f"Data sources configuration for {config_type} '{config}' in "
                f"cause '{cause_name}' must be a dictionary if it is present."
            )
            return error_messages

        if not set(data_sources_config.keys()).issubset(self._DATA_SOURCE_KEYS[config_type]):
            error_messages.append(
                f"Data sources configuration for {config_type} '{config_name}' "
                f"in cause '{cause_name}' may only contain the following keys: "
                f"{self._DATA_SOURCE_KEYS[config_type]}."
            )

        for config_name, source in data_sources_config.items():
            try:
                self._get_data_source(config_name, source)
            except ValueError:
                error_messages.append(
                    f"Configuration for {config_type} '{config_name}' in cause "
                    f"'{cause_name}' has an invalid data source at '{source}'."
                )
        return error_messages
