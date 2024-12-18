"""
===================
Disease Transitions
===================

This module contains tools to model transitions between disease states.

"""
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.state_machine import Transition, Trigger
from vivarium.framework.utilities import rate_to_probability
from vivarium.framework.values import list_combiner, union_post_processor
from vivarium.types import DataInput

from vivarium_public_health.disease.exceptions import DiseaseModelError
from vivarium_public_health.utilities import get_lookup_columns

if TYPE_CHECKING:
    from vivarium_public_health.disease import BaseDiseaseState


class TransitionString(str):
    def __new__(cls, value):
        # noinspection PyArgumentList
        obj = str.__new__(cls, value.lower())
        obj.from_state, obj.to_state = value.split("_TO_")
        return obj

    def __getnewargs__(self):
        return (self.from_state + "_TO_" + self.to_state,)


class RateTransition(Transition):

    ##############
    # Properties #
    ##############

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        return {
            f"{self.name}": {
                "data_sources": {
                    "transition_rate": self._rate_source,
                },
            },
        }

    @property
    def columns_required(self) -> list[str] | None:
        return ["alive"]

    @property
    def transition_rate_pipeline_name(self) -> str:
        if self._get_data_functions:
            if "incidence_rate" in self._get_data_functions:
                pipeline_name = f"{self.output_state.state_id}.incidence_rate"
            elif "remission_rate" in self._get_data_functions:
                pipeline_name = f"{self.input_state.state_id}.remission_rate"
            elif "transition_rate" in self._get_data_functions:
                pipeline_name = (
                    f"{self.input_state.state_id}_to_{self.output_state.state_id}"
                    ".transition_rate"
                )
            else:
                raise DiseaseModelError(
                    "Cannot determine rate_transition pipeline name: "
                    "no valid data functions supplied."
                )
        else:
            if self.rate_type == "incidence_rate":
                pipeline_name = f"{self.output_state.state_id}.incidence_rate"
            elif self.rate_type == "remission_rate":
                pipeline_name = f"{self.input_state.state_id}.remission_rate"
            elif self.rate_type == "transition_rate":
                pipeline_name = (
                    f"{self.input_state.state_id}_to_{self.output_state.state_id}"
                    ".transition_rate"
                )
            else:
                raise DiseaseModelError(
                    "Cannot determine rate_transition pipeline name: invalid"
                    f" rate_type '{self.rate_type} supplied."
                )

        return pipeline_name

    #####################
    # Lifecycle methods #
    #####################

    def __init__(
        self,
        input_state: "BaseDiseaseState",
        output_state: "BaseDiseaseState",
        get_data_functions: dict[str, Callable] = None,
        triggered=Trigger.NOT_TRIGGERED,
        transition_rate: DataInput | None = None,
        rate_type: str = "transition_rate",
    ):
        super().__init__(
            input_state, output_state, probability_func=self._probability, triggered=triggered
        )
        self._get_data_functions = (
            get_data_functions if get_data_functions is not None else {}
        )
        self._rate_source = self._get_rate_source(transition_rate)
        self.rate_type = rate_type

        if get_data_functions is not None:
            warnings.warn(
                "The argument 'get_data_functions' has been deprecated. Use"
                " 'transition_rate' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if transition_rate is not None:
                raise DiseaseModelError(
                    "It is not allowed to pass a transition rate"
                    " both as a stand-alone argument and as part of"
                    " get_data_functions."
                )

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        lookup_columns = get_lookup_columns([self.lookup_tables["transition_rate"]])
        self.transition_rate = builder.value.register_rate_producer(
            self.transition_rate_pipeline_name,
            source=self.compute_transition_rate,
            requires_columns=lookup_columns + ["alive"],
            requires_values=[f"{self.transition_rate_pipeline_name}.paf"],
        )
        paf = builder.lookup.build_table(0)
        self.joint_paf = builder.value.register_value_producer(
            f"{self.transition_rate_pipeline_name}.paf",
            source=lambda index: [paf(index)],
            preferred_combiner=list_combiner,
            preferred_post_processor=union_post_processor,
        )

    #################
    # Setup methods #
    #################

    def _get_rate_source(self, transition_rate: DataInput | None) -> DataInput:
        if transition_rate is not None:
            rate_data = transition_rate
        elif "incidence_rate" in self._get_data_functions:
            rate_data = lambda builder: self._get_data_functions["incidence_rate"](
                builder, self.output_state.state_id
            )
        elif "remission_rate" in self._get_data_functions:
            rate_data = lambda builder: self._get_data_functions["remission_rate"](
                builder, self.input_state.state_id
            )
        elif "transition_rate" in self._get_data_functions:
            rate_data = lambda builder: self._get_data_functions["transition_rate"](
                builder, self.input_state.state_id, self.output_state.state_id
            )
        else:
            raise ValueError("No valid data functions supplied.")
        return rate_data

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def compute_transition_rate(self, index: pd.Index) -> pd.Series:
        transition_rate = pd.Series(0.0, index=index)
        living = self.population_view.get(index, query='alive == "alive"').index
        base_rates = self.lookup_tables["transition_rate"](living)
        joint_paf = self.joint_paf(living)
        transition_rate.loc[living] = base_rates * (1 - joint_paf)
        return transition_rate

    ##################
    # Helper methods #
    ##################

    def _probability(self, index: pd.Index) -> pd.Series:
        return pd.Series(rate_to_probability(self.transition_rate(index)))


class ProportionTransition(Transition):

    ##############
    # Properties #
    ##############

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        return {
            f"{self.name}": {
                "data_sources": {
                    "proportion": self.load_proportion,
                },
            },
        }

    #####################
    # Lifecycle methods #
    #####################

    def __init__(
        self,
        input_state: "BaseDiseaseState",
        output_state: "BaseDiseaseState",
        get_data_functions: dict[str, Callable] = None,
        triggered=Trigger.NOT_TRIGGERED,
        proportion: DataInput | None = None,
    ):
        super().__init__(
            input_state, output_state, probability_func=self._probability, triggered=triggered
        )
        self._proportion_source = proportion
        self._get_data_functions = (
            get_data_functions if get_data_functions is not None else {}
        )

        if get_data_functions is not None:
            warnings.warn(
                "The argument 'get_data_functions' has been deprecated. Use"
                " 'proportion' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if proportion is not None:
                raise DiseaseModelError(
                    "It is not allowed to pass a proportion both as a"
                    " stand-alone argument and as part of get_data_functions."
                )

    #################
    # Setup methods #
    #################

    def load_proportion(self, builder: Builder) -> DataInput:
        if self._proportion_source is not None:
            return self._proportion_source
        if "proportion" not in self._get_data_functions:
            raise DiseaseModelError("Must supply a proportion function")
        return self._get_data_functions["proportion"](builder, self.output_state.state_id)

    def _probability(self, index):
        return self.lookup_tables["proportion"](index)
