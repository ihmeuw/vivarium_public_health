"""
=================
The Disease Model
=================

This module contains a state machine driver for disease models.  Its primary
function is to provide coordination across a set of disease states and
transitions at simulation initialization and during transitions.

"""
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from vivarium.exceptions import VivariumError
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.state_machine import Machine

from vivarium_public_health.disease.state import BaseDiseaseState, SusceptibleState
from vivarium_public_health.disease.transition import TransitionString


class DiseaseModelError(VivariumError):
    pass


class DiseaseModel(Machine):
    ##############
    # Properties #
    ##############

    @property
    def columns_created(self) -> List[str]:
        return [self.state_column]

    @property
    def columns_required(self) -> Optional[List[str]]:
        return ["age", "sex"]

    @property
    def initialization_requirements(self) -> Dict[str, List[str]]:
        return {
            "requires_columns": ["age", "sex"],
            "requires_values": [],
            "requires_streams": [f"{self.state_column}_initial_states"],
        }

    @property
    def state_names(self) -> List[str]:
        return [s.state_id for s in self.states]

    @property
    def transition_names(self) -> List[TransitionString]:
        return [
            state_name for state in self.states for state_name in state.get_transition_names()
        ]

    #####################
    # Lifecycle methods #
    #####################

    def __init__(
        self,
        cause: str,
        initial_state: Optional[BaseDiseaseState] = None,
        get_data_functions: Optional[Dict[str, Callable]] = None,
        cause_type: str = "cause",
        **kwargs,
    ):
        super().__init__(cause, **kwargs)
        self.cause = cause
        self.cause_type = cause_type

        if initial_state is not None:
            self.initial_state = initial_state.state_id
        else:
            self.initial_state = self._get_default_initial_state()

        self._get_data_functions = (
            get_data_functions if get_data_functions is not None else {}
        )

    def setup(self, builder: Builder) -> None:
        """Perform this component's setup."""
        super().setup(builder)

        self.configuration_age_start = builder.configuration.population.initialization_age_min
        self.configuration_age_end = builder.configuration.population.initialization_age_max

        cause_specific_mortality_rate = self.load_cause_specific_mortality_rate_data(builder)
        self.cause_specific_mortality_rate = builder.lookup.build_table(
            cause_specific_mortality_rate,
            key_columns=["sex"],
            parameter_columns=["age", "year"],
        )
        builder.value.register_value_modifier(
            "cause_specific_mortality_rate",
            self.adjust_cause_specific_mortality_rate,
            requires_columns=["age", "sex"],
        )
        self.randomness = builder.randomness.get_stream(f"{self.state_column}_initial_states")

    #################
    # Setup methods #
    #################

    def load_cause_specific_mortality_rate_data(self, builder):
        if "cause_specific_mortality_rate" not in self._get_data_functions:
            only_morbid = builder.data.load(f"cause.{self.cause}.restrictions")["yld_only"]
            if only_morbid:
                csmr_data = 0
            else:
                csmr_data = builder.data.load(
                    f"{self.cause_type}.{self.cause}.cause_specific_mortality_rate"
                )
        else:
            csmr_data = self._get_data_functions["cause_specific_mortality_rate"](
                self.cause, builder
            )
        return csmr_data

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        population = self.population_view.subview(["age", "sex"]).get(pop_data.index)

        assert self.initial_state in {s.state_id for s in self.states}

        # FIXME: this is a hack to figure out whether or not we're at the simulation start based on the fact that the
        #  fertility components create this user data
        if pop_data.user_data["sim_state"] == "setup":  # simulation start
            if self.configuration_age_start != self.configuration_age_end != 0:
                state_names, weights_bins = self.get_state_weights(
                    pop_data.index, "prevalence"
                )
            else:
                raise NotImplementedError(
                    "We do not currently support an age 0 cohort. "
                    "configuration.population.initialization_age_min and "
                    "configuration.population.initialization_age_max "
                    "cannot both be 0."
                )

        else:  # on time step
            if pop_data.user_data["age_start"] == pop_data.user_data["age_end"] == 0:
                state_names, weights_bins = self.get_state_weights(
                    pop_data.index, "birth_prevalence"
                )
            else:
                state_names, weights_bins = self.get_state_weights(
                    pop_data.index, "prevalence"
                )

        if state_names and not population.empty:
            # only do this if there are states in the model that supply prevalence data
            population["sex_id"] = population.sex.apply({"Male": 1, "Female": 2}.get)

            condition_column = self.assign_initial_status_to_simulants(
                population,
                state_names,
                weights_bins,
                self.randomness.get_draw(population.index),
            )

            condition_column = condition_column.rename(
                columns={"condition_state": self.state_column}
            )
        else:
            condition_column = pd.Series(
                self.initial_state, index=population.index, name=self.state_column
            )
        self.population_view.update(condition_column)

    def on_time_step(self, event: Event) -> None:
        self.transition(event.index, event.time)

    def on_time_step_cleanup(self, event: Event) -> None:
        self.cleanup(event.index, event.time)

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def adjust_cause_specific_mortality_rate(self, index, rate):
        return rate + self.cause_specific_mortality_rate(index)

    ####################
    # Helper functions #
    ####################

    def _get_default_initial_state(self):
        susceptible_states = [s for s in self.states if isinstance(s, SusceptibleState)]
        if len(susceptible_states) != 1:
            raise DiseaseModelError("Disease model must have exactly one SusceptibleState.")
        return susceptible_states[0].state_id

    def get_state_weights(self, pop_index, prevalence_type):
        states = [
            s
            for s in self.states
            if hasattr(s, f"{prevalence_type}")
            and getattr(s, f"{prevalence_type}") is not None
        ]

        if not states:
            return states, None

        weights = [getattr(s, f"{prevalence_type}")(pop_index) for s in states]
        for w in weights:
            w.reset_index(inplace=True, drop=True)
        weights += ((1 - np.sum(weights, axis=0)),)

        weights = np.array(weights).T
        weights_bins = np.cumsum(weights, axis=1)

        state_names = [s.state_id for s in states] + [self.initial_state]

        return state_names, weights_bins

    @staticmethod
    def assign_initial_status_to_simulants(
        simulants_df, state_names, weights_bins, propensities
    ):
        simulants = simulants_df[["age", "sex"]].copy()

        choice_index = (propensities.values[np.newaxis].T > weights_bins).sum(axis=1)
        initial_states = pd.Series(np.array(state_names)[choice_index], index=simulants.index)

        simulants.loc[:, "condition_state"] = initial_states
        return simulants
