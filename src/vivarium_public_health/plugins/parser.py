from importlib import import_module
from typing import Any, Callable, Dict, List, Union

import pandas as pd
from pkg_resources import resource_filename
from vivarium import Component, ConfigTree
from vivarium.framework.components import ComponentConfigurationParser
from vivarium.framework.engine import Builder
from vivarium.framework.state_machine import Trigger
from vivarium_public_health.disease import (
    BaseDiseaseState,
    DiseaseModel,
    DiseaseState,
    RecoveredState,
    SusceptibleState,
    TransientDiseaseState,
)
from vivarium_public_health.utilities import TargetString

from vivarium_nih_us_cvd.components.causes.state import (
    MultiTransitionDiseaseState,
    MultiTransitionSusceptibleState,
)


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
        It also marks states that have multiple exiting transitions with the
        `is_multi_transition` key in the `model_override` layer.

        Parameters
        ----------
        component_config
            A ConfigTree defining the components to initialize.

        Returns
        -------
        List[Component]
            A list of initialized components.
        """
        components = []

        if not component_config:
            return components

        if "external_configuration" in component_config:
            for package, config_files in component_config["external_configuration"].items():
                for config_file in config_files.get_value(None):
                    source = f"{package}::{config_file}"
                    config_file = resource_filename(package, config_file)

                    external_config = ConfigTree(config_file)
                    component_config.update(
                        external_config, layer="model_override", source=source
                    )

        if "causes" in component_config:
            self.mark_multi_transition_states(component_config)
            self.add_default_config_layer(component_config)
            components += self.get_cause_model_components(component_config["causes"])

        # Create a copy of the config tree excluding "external_config" and
        # "causes" keys.
        component_config_dict = component_config.to_dict()
        component_config_dict.pop("external_configuration", None)
        component_config_dict.pop("causes", None)

        components += self.process_level(component_config_dict, [])
        return components

    ##################################
    # Configuration creation methods #
    ##################################

    @staticmethod
    def mark_multi_transition_states(component_config: ConfigTree) -> None:
        """
        Marks states that have multiple exiting transitions using the
        `is_multi_transition` key.

        Parameters
        ----------
        component_config
            A ConfigTree defining the components to initialize

        Returns
        -------
        None
        """
        transition_counts = {
            cause: {state: 0 for state in config.states}
            for cause, config in component_config.causes.items()
        }

        for cause, config in component_config.causes.items():
            for transition in config.transitions.values():
                transition_counts[cause][transition.source] += 1

        for cause, states in transition_counts.items():
            for state, counts in states.items():
                component_config.causes[cause].states[state].update(
                    {"is_multi_transition": counts > 1},
                    layer="model_override",
                    source="causes_configuration_parser",
                )

    @staticmethod
    def add_default_config_layer(component_config: ConfigTree) -> None:
        """
        Adds a default layer to the provided configuration that specifies
        default values for the cause model configuration.

        Parameters
        ----------
        component_config
            A ConfigTree that specifies the components to initialize

        Returns
        -------
        None
        """
        default_config = {"causes": {}}
        for cause_name, cause_config in component_config.causes.items():
            default_states_config = {}
            default_transitions_config = {}
            default_config["causes"][cause_name] = {
                "states": default_states_config,
                "transitions": default_transitions_config,
            }

            for state_name, state_config in cause_config.states.items():
                default_states_config[state_name] = {
                    "cause_type": "cause",
                    "is_multi_transition": False,
                    "transient": False,
                    "allow_self_transition": True,
                    "side_effect": None,
                    "cleanup_function": None,
                }

            for transition_name, transition_config in cause_config.transitions.items():
                default_transitions_config[transition_name] = {"triggered": "NOT_TRIGGERED"}

        component_config.update(
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

            cause_models.append(DiseaseModel(cause_name, states=list(states.values())))

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
        if "get_data_functions" in state_config:
            data_getters_config = state_config.get_data_functions
            state_kwargs["get_data_functions"] = {
                name: self.get_data_getter(name, data_getters_config[name])
                for name in data_getters_config.keys()
            }

        if state_config.transient:
            state_type = TransientDiseaseState
        elif state_config.is_multi_transition and state_name == "susceptible":
            state_type = MultiTransitionSusceptibleState
        elif state_config.is_multi_transition:
            state_type = MultiTransitionDiseaseState
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
        if "get_data_functions" in transition_config:
            data_getters_config = transition_config.get_data_functions
            data_getters = {
                name: self.get_data_getter(name, data_getters_config[name])
                for name in data_getters_config.keys()
            }
        else:
            data_getters = None

        if transition_config.data_type == "rate":
            source_state.add_rate_transition(
                sink_state, get_data_functions=data_getters, triggered=triggered
            )
        elif transition_config.data_type == "proportion":
            source_state.add_proportion_transition(
                sink_state, get_data_functions=data_getters, triggered=triggered
            )
        elif transition_config.data_type == "dwell_time":
            source_state.add_dwell_time_transition(sink_state, triggered=triggered)
        else:
            raise ValueError(
                f"Invalid transition data type '{transition_config.data_type}'"
                f" provided for transition '{transition_config}'."
            )

    @staticmethod
    def get_data_getter(
        name: str, getter: Union[str, float]
    ) -> Callable[[Builder, Any], Any]:
        """
        Parses a data getter and returns a callable that can be used to retrieve
        the data.

        Parameters
        ----------
        name
            The name of the data getter
        getter
            The data getter to parse

        Returns
        -------
        Callable[[Builder, Any], Any]
            A callable that can be used to retrieve the data
        """
        if isinstance(getter, float):
            return lambda builder, *_: getter

        try:
            timedelta = pd.Timedelta(getter)
            return lambda builder, *_: timedelta
        except ValueError:
            pass

        if "::" in getter:
            module, method = getter.split("::")
            return getattr(import_module(module), method)

        try:
            target_string = TargetString(getter)
            return lambda builder, *_: builder.data.load(target_string)
        except ValueError:
            pass

        raise ValueError(f"Invalid data getter '{getter}' for '{name}'.")
