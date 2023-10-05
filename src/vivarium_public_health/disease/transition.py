"""
===================
Disease Transitions
===================

This module contains tools to model transitions between disease states.

"""
from typing import TYPE_CHECKING, Callable, Dict

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.state_machine import Transition, Trigger
from vivarium.framework.utilities import rate_to_probability
from vivarium.framework.values import list_combiner, union_post_processor

if TYPE_CHECKING:
    from vivarium_public_health.disease import BaseDiseaseState


class TransitionString(str):
    def __new__(cls, value):
        # noinspection PyArgumentList
        obj = str.__new__(cls, value.lower())
        obj.from_state, obj.to_state = value.split("_TO_")
        return obj


class RateTransition(Transition):
    #####################
    # Lifecycle methods #
    #####################

    def __init__(
        self,
        input_state: "BaseDiseaseState",
        output_state: "BaseDiseaseState",
        get_data_functions: Dict[str, Callable] = None,
        triggered=Trigger.NOT_TRIGGERED,
    ):
        super().__init__(
            input_state, output_state, probability_func=self._probability, triggered=triggered
        )
        self._get_data_functions = (
            get_data_functions if get_data_functions is not None else {}
        )

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        rate_data, pipeline_name = self.load_transition_rate_data(builder)
        self.base_rate = builder.lookup.build_table(
            rate_data, key_columns=["sex"], parameter_columns=["age", "year"]
        )
        self.transition_rate = builder.value.register_rate_producer(
            pipeline_name,
            source=self.compute_transition_rate,
            requires_columns=["age", "sex", "alive"],
            requires_values=[f"{pipeline_name}.paf"],
        )
        paf = builder.lookup.build_table(0)
        self.joint_paf = builder.value.register_value_producer(
            f"{pipeline_name}.paf",
            source=lambda index: [paf(index)],
            preferred_combiner=list_combiner,
            preferred_post_processor=union_post_processor,
        )

        self.population_view = builder.population.get_view(["alive"])

    #################
    # Setup methods #
    #################

    def load_transition_rate_data(self, builder):
        if "incidence_rate" in self._get_data_functions:
            rate_data = self._get_data_functions["incidence_rate"](
                builder, self.output_state.state_id
            )
            pipeline_name = f"{self.output_state.state_id}.incidence_rate"
        elif "remission_rate" in self._get_data_functions:
            rate_data = self._get_data_functions["remission_rate"](
                builder, self.input_state.state_id
            )
            pipeline_name = f"{self.input_state.state_id}.remission_rate"
        elif "transition_rate" in self._get_data_functions:
            rate_data = self._get_data_functions["transition_rate"](
                builder, self.input_state.state_id, self.output_state.state_id
            )
            pipeline_name = (
                f"{self.input_state.state_id}_to_{self.output_state.state_id}.transition_rate"
            )
        else:
            raise ValueError("No valid data functions supplied.")
        return rate_data, pipeline_name

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def compute_transition_rate(self, index: pd.Index) -> pd.Series:
        transition_rate = pd.Series(0.0, index=index)
        living = self.population_view.get(index, query='alive == "alive"').index
        base_rates = self.base_rate(living)
        joint_paf = self.joint_paf(living)
        transition_rate.loc[living] = base_rates * (1 - joint_paf)
        return transition_rate

    ##################
    # Helper methods #
    ##################

    def _probability(self, index: pd.Index) -> pd.Series:
        return pd.Series(rate_to_probability(self.transition_rate(index)))


class ProportionTransition(Transition):
    #####################
    # Lifecycle methods #
    #####################

    def __init__(
        self,
        input_state: "BaseDiseaseState",
        output_state: "BaseDiseaseState",
        get_data_functions: Dict[str, Callable] = None,
        triggered=Trigger.NOT_TRIGGERED,
    ):
        super().__init__(
            input_state, output_state, probability_func=self._probability, triggered=triggered
        )
        self._get_data_functions = (
            get_data_functions if get_data_functions is not None else {}
        )

    # noinspection PyAttributeOutsideInit
    def setup(self, builder):
        super().setup(builder)
        get_proportion_func = self._get_data_functions.get("proportion", None)
        if get_proportion_func is None:
            raise ValueError("Must supply a proportion function")
        self._proportion_data = get_proportion_func(builder, self.output_state.state_id)
        self.proportion = builder.lookup.build_table(
            self._proportion_data, key_columns=["sex"], parameter_columns=["age", "year"]
        )

    def _probability(self, index):
        return self.proportion(index)
