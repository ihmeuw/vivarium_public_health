"""
===================
Disability Observer
===================

This module contains tools for observing years lived with disability (YLDs)
in the simulation.

"""
from typing import List

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.values import (
    NumberLike,
    Pipeline,
    list_combiner,
    rescale_post_processor,
    union_post_processor,
)

from vivarium_public_health.disease import DiseaseState, RiskAttributableDisease
from vivarium_public_health.utilities import to_years


class DisabilityObserver:
    """Counts years lived with disability.

    By default, this counts both aggregate and cause-specific years lived
    with disability over the full course of the simulation.

    In the model specification, your configuration for this component should
    be specified as, e.g.:

    .. code-block:: yaml

        configuration:
            observers:
                disability:
                    exclude:
                        - "sex"
                    include:
                        - "sample_stratification"
    """

    configuration_defaults = {
        "stratification": {
            "disability": {
                "exclude": [],
                "include": [],
            }
        }
    }

    def __init__(self):
        # self.ylds_column_name = "years_lived_with_disability"
        # self.metrics_pipeline_name = "metrics"
        self.disability_weight_pipeline_name = "disability_weight"

    def __repr__(self):
        return "DisabilityObserver()"

    @property
    def name(self):
        return "disability_observer"

    @property
    def disease_classes(self) -> List:
        return [DiseaseState, RiskAttributableDisease]

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder):
        self.config = builder.configuration.stratification.disability
        self.step_size = pd.Timedelta(days=builder.configuration.time.step_size)
        self.disability_weight = self._get_disability_weight_pipeline(builder)
        cause_states = builder.components.get_components_by_type(tuple(self.disease_classes))

        builder.results.register_observation(
            name="ylds_due_to_all_causes",
            pop_filter='tracked == True and alive == "alive"',
            aggregator_sources=[self.disability_weight_pipeline_name],
            aggregator=self._disability_weight_aggregator,
            requires_columns=["alive"],
            requires_values=["disability_weight"],
            additional_stratifications=self.config.include,
            excluded_stratifications=self.config.exclude,
            when="time_step__prepare",
        )

        for cause_state in cause_states:
            cause_disability_weight_pipeline_name = f"{cause_state.state_id}.disability_weight"
            builder.results.register_observation(
                name=f"ylds_due_to_{cause_state.state_id}",
                pop_filter='tracked == True and alive == "alive"',
                aggregator_sources=[cause_disability_weight_pipeline_name],
                aggregator=self._disability_weight_aggregator,
                requires_columns=["alive"],
                requires_values=[cause_disability_weight_pipeline_name],
                additional_stratifications=self.config.include,
                excluded_stratifications=self.config.exclude,
                when="time_step__prepare",
            )

    def _get_disability_weight_pipeline(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.disability_weight_pipeline_name,
            source=lambda index: [pd.Series(0.0, index=index)],
            preferred_combiner=list_combiner,
            preferred_post_processor=self._disability_post_processor,
        )

    def _disability_weight_aggregator(self, dw: pd.DataFrame) -> float:
        return (dw * to_years(self.step_size)).sum().squeeze()

    def _disability_post_processor(self, value: NumberLike, step_size: pd.Timedelta) -> NumberLike:
        return rescale_post_processor(union_post_processor(value, self.step_size), self.step_size)
