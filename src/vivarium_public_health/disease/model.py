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
from layered_config_tree import ConfigurationError
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.state_machine import Machine
from vivarium.types import DataInput, LookupTableData

from vivarium_public_health.disease.exceptions import DiseaseModelError
from vivarium_public_health.disease.state import BaseDiseaseState, SusceptibleState
from vivarium_public_health.disease.transition import RateTransition, TransitionString


class DiseaseModel(Machine):

    ##############
    # Properties #
    ##############

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        """Provides default configuration values for this disease model.

        Configuration structure::

            {disease_name}:
                data_sources:
                    cause_specific_mortality_rate:
                        Source for cause-specific mortality rate (CSMR) data.
                        Default uses the ``load_cause_specific_mortality_rate``
                        method which loads from artifact at
                        ``cause.{cause_name}.cause_specific_mortality_rate``.
        """
        return {
            f"{self.name}": {
                "data_sources": {
                    "cause_specific_mortality_rate": self.load_cause_specific_mortality_rate,
                },
            },
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
        cause_type: str = "cause",
        states: Iterable[BaseDiseaseState] = (),
        residual_state: BaseDiseaseState | None = None,
        cause_specific_mortality_rate: DataInput | None = None,
    ) -> None:
        super().__init__(cause, states=states)
        self.cause = cause
        self.cause_type = cause_type
        self.residual_state = self._get_residual_state(residual_state)
        self._csmr_source = cause_specific_mortality_rate

    def setup(self, builder: Builder) -> None:
        """Perform this component's setup."""
        self.initialization_weights_pipelines = [
            *[state.prevalence_pipeline for state in self.states],
            *[state.birth_prevalence_pipeline for state in self.states],
        ]
        super().setup(builder)

        self.configuration_age_start = builder.configuration.population.initialization_age_min
        self.configuration_age_end = builder.configuration.population.initialization_age_max

        self.csmr_table = self.build_lookup_table(builder, "cause_specific_mortality_rate")

        builder.value.register_attribute_modifier(
            "cause_specific_mortality_rate",
            self.adjust_cause_specific_mortality_rate,
            required_resources=["age", "sex"],
        )

    def on_post_setup(self, event: Event) -> None:
        conversion_types = set()
        for state in self.states:
            for transition in state.transition_set.transitions:
                if isinstance(transition, RateTransition):
                    conversion_types.add(transition.rate_conversion_type)
        if len(conversion_types) > 1:
            raise ConfigurationError(
                "All transitions in a disease model must have the same rate conversion type."
                f" Found: {conversion_types}."
            )
        # TODO validate that all states which have a transition of type Transition
        #  (note: don't use an isinstance check here) have a non-zero dwell time, are transient
        #  states, or don't allow self-transitions.

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        """Initialize the simulants in the population.

        If all simulants are initialized at age 0, birth prevalence is used.
        Otherwise, prevalence is used.

        Parameters
        ----------
        pop_data
            The population data object.
        """

        self.initialization_weights_pipelines = [
            state.birth_prevalence_pipeline
            if pop_data.user_data.get("age_end", self.configuration_age_end) == 0
            else state.prevalence_pipeline
            for state in self.states
        ]

        super().on_initialize_simulants(pop_data)

    #################
    # Setup methods #
    #################

    def load_cause_specific_mortality_rate(self, builder: Builder) -> float | pd.DataFrame:
        if self._csmr_source is None:
            only_morbid = builder.data.load(f"cause.{self.cause}.restrictions")["yld_only"]
            if only_morbid:
                self._csmr_source = 0.0
            else:
                self._csmr_source = (
                    f"{self.cause_type}.{self.cause}.cause_specific_mortality_rate"
                )
        return self.get_data(builder, self._csmr_source)

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def adjust_cause_specific_mortality_rate(self, index, rate):
        return rate + self.csmr_table(index)

    ####################
    # Helper functions #
    ####################

    def _get_residual_state(self, residual_state: BaseDiseaseState) -> BaseDiseaseState:
        """Get the residual state for the DiseaseModel.

        This will be the residual state if it is provided, otherwise it will be
        the model's SusceptibleState. This method also calculates the residual
        state's birth_prevalence and prevalence.
        """
        if residual_state is None:
            susceptible_states = [s for s in self.states if isinstance(s, SusceptibleState)]
            if len(susceptible_states) != 1:
                raise DiseaseModelError(
                    "DiseaseModel must have exactly one SusceptibleState or it must specify"
                    " a residual state."
                )
            residual_state = susceptible_states[0]

        if residual_state not in self.states:
            raise DiseaseModelError(
                f"Residual state '{residual_state}' must be one of the states: {self.states}."
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
