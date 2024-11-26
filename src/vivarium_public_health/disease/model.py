"""
=================
The Disease Model
=================

This module contains a state machine driver for disease models.  Its primary
function is to provide coordination across a set of disease states and
transitions at simulation initialization and during transitions.

"""
import warnings
from collections.abc import Callable, Iterable
from functools import partial
from typing import Any

import pandas as pd
from vivarium.exceptions import VivariumError
from vivarium.framework.engine import Builder
from vivarium.framework.population import SimulantData
from vivarium.framework.state_machine import Machine
from vivarium.types import LookupTableData

from vivarium_public_health.disease.state import BaseDiseaseState, SusceptibleState
from vivarium_public_health.disease.transition import TransitionString


class DiseaseModelError(VivariumError):
    pass


class DiseaseModel(Machine):

    ##############
    # Properties #
    ##############

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        return {
            f"{self.name}": {
                "data_sources": {
                    "cause_specific_mortality_rate": self.load_cause_specific_mortality_rate,
                },
            },
        }

    @property
    def columns_created(self) -> list[str]:
        return [self.state_column]

    @property
    def columns_required(self) -> list[str] | None:
        return ["age", "sex"]

    @property
    def initialization_requirements(self) -> dict[str, list[str]]:
        return {
            "requires_columns": ["age", "sex"],
            "requires_values": [],
            "requires_streams": [f"{self.state_column}_initial_states"],
        }

    @property
    def state_names(self) -> list[str]:
        return [s.state_id for s in self.states]

    @property
    def transition_names(self) -> list[TransitionString]:
        return [
            state_name for state in self.states for state_name in state.get_transition_names()
        ]

    #####################
    # Lifecycle methods #
    #####################

    def __init__(
        self,
        cause: str,
        initial_state: BaseDiseaseState | None = None,
        get_data_functions: dict[str, Callable] | None = None,
        cause_type: str = "cause",
        states: Iterable[BaseDiseaseState] = (),
        residual_state: BaseDiseaseState | None = None,
    ):
        super().__init__(cause, states=states)
        self.cause = cause
        self.cause_type = cause_type
        self.residual_state = self._get_residual_state(initial_state, residual_state)
        self._get_data_functions = (
            get_data_functions if get_data_functions is not None else {}
        )

    def setup(self, builder: Builder) -> None:
        """Perform this component's setup."""
        super().setup(builder)

        self.configuration_age_start = builder.configuration.population.initialization_age_min
        self.configuration_age_end = builder.configuration.population.initialization_age_max

        builder.value.register_value_modifier(
            "cause_specific_mortality_rate",
            self.adjust_cause_specific_mortality_rate,
            requires_columns=["age", "sex"],
        )

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        """Initialize the simulants in the population.

        If all simulants are initialized at age 0, birth prevalence is used.
        Otherwise, prevalence is used.

        Parameters
        ----------
        pop_data
            The population data object.
        """
        if pop_data.user_data.get("age_end", self.configuration_age_end) == 0:
            initialization_table_name = "birth_prevalence"
        else:
            initialization_table_name = "prevalence"

        for state in self.states:
            state.lookup_tables["initialization_weights"] = state.lookup_tables[
                initialization_table_name
            ]

        super().on_initialize_simulants(pop_data)

    #################
    # Setup methods #
    #################

    def load_cause_specific_mortality_rate(self, builder: Builder) -> float | pd.DataFrame:
        if "cause_specific_mortality_rate" not in self._get_data_functions:
            only_morbid = builder.data.load(f"cause.{self.cause}.restrictions")["yld_only"]
            if only_morbid:
                csmr_data = 0.0
            else:
                csmr_data = builder.data.load(
                    f"{self.cause_type}.{self.cause}.cause_specific_mortality_rate"
                )
        else:
            csmr_data = self._get_data_functions["cause_specific_mortality_rate"](
                self.cause, builder
            )
        return csmr_data

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def adjust_cause_specific_mortality_rate(self, index, rate):
        return rate + self.lookup_tables["cause_specific_mortality_rate"](index)

    ####################
    # Helper functions #
    ####################

    def _get_residual_state(
        self, initial_state: BaseDiseaseState, residual_state: BaseDiseaseState
    ) -> BaseDiseaseState:
        """Get the residual state for the DiseaseModel.

        This will be the residual state if it is provided, otherwise it will be
        the model's SusceptibleState. This method also calculates the residual
        state's birth_prevalence and prevalence.
        """
        if initial_state is not None:
            warnings.warn(
                "In the future, the 'initial_state' argument to DiseaseModel"
                " will be used to initialize all simulants into that state. To"
                " retain the current behavior of defining a residual state, use"
                " the 'residual_state' argument.",
                DeprecationWarning,
                stacklevel=2,
            )

            if residual_state:
                raise DiseaseModelError(
                    "A DiseaseModel cannot be initialized with both"
                    " 'initial_state and 'residual_state'."
                )

            residual_state = initial_state
        elif residual_state is None:
            susceptible_states = [s for s in self.states if isinstance(s, SusceptibleState)]
            if len(susceptible_states) != 1:
                raise DiseaseModelError(
                    "Disease model must have exactly one SusceptibleState."
                )
            residual_state = susceptible_states[0]

        if residual_state not in self.states:
            raise DiseaseModelError(
                f"Residual state '{self.residual_state}' must be one of the"
                f" states: {self.states}."
            )

        residual_state.birth_prevalence = partial(
            self._get_residual_state_probabilities, table_name="birth_prevalence"
        )
        residual_state.prevalence = partial(
            self._get_residual_state_probabilities, table_name="prevalence"
        )

        return residual_state

    def _get_residual_state_probabilities(
        self, builder: Builder, table_name: str
    ) -> LookupTableData:
        """Calculate the probabilities of the residual state based on the other states."""
        non_residual_states = [s for s in self.states if s != self.residual_state]
        non_residual_probabilities = 0
        for state in non_residual_states:
            weights_source = builder.configuration[state.name].data_sources[table_name]
            weights = state.get_data(builder, weights_source)
            if isinstance(weights, pd.DataFrame):
                weights = weights.set_index(
                    [c for c in weights.columns if c != "value"]
                ).squeeze()
            non_residual_probabilities += weights

        residual_probabilities = 1 - non_residual_probabilities

        if pd.Series(residual_probabilities < 0).any():
            raise ValueError(
                f"The {table_name} for the states in the DiseaseModel must sum"
                " to less than 1."
            )
        if isinstance(residual_probabilities, pd.Series):
            residual_probabilities = residual_probabilities.reset_index()

        return residual_probabilities
