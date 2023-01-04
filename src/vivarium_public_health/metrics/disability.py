"""
===================
Disability Observer
===================

This module contains tools for observing years lived with disability (YLDs)
in the simulation.

"""
from collections import Counter
from typing import Dict, List, Union

import pandas as pd
from vivarium.config_tree import ConfigTree
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import PopulationView
from vivarium.framework.values import (
    NumberLike,
    Pipeline,
    list_combiner,
    rescale_post_processor,
    union_post_processor,
)

from vivarium_public_health.disease import DiseaseState, RiskAttributableDisease
from vivarium_public_health.metrics.stratification import ResultsStratifier
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

    @property
    def disease_classes(self) -> List:
        return [DiseaseState, RiskAttributableDisease]

    def setup(self, builder: Builder):
        self.config = builder.configuration.stratification.disability
        self.disability_weight = builder.value.get_value("disability_weight")
        cause_states = builder.components.get_components_by_type(tuple(self.disease_classes))

        builder.results.register_observation(
            name="ylds_due_to_all_causes",
            pop_filter='tracked == True and alive == "alive"',
            aggregator_sources=[str(self.disability_weight)],
            aggregator=self._disability_weight_aggregator,
            requires_columns=["alive"],
            requires_values=["disability_weight"],
            additional_stratifications=self.config.include,
            excluded_stratifications=self.config.exclude,
            when="time_step__prepare",
        )

        for cause_state in cause_states:
            pipeline = builder.value.get_value(f"{cause_state.state_id}.disability_weight")
            builder.results.register_observation(
                name=f"ylds_due_to_{cause_state.state_id}",
                pop_filter='tracked == True and alive == "alive"',
                aggregator_sources=[str(pipeline)],
                aggregator=self._disability_weight_aggregator,
                requires_columns=["alive"],
                requires_values=[f"{cause_state.state_id}.disability_weight"],
                additional_stratifications=self.config.include,
                excluded_stratifications=self.config.exclude,
                when="time_step__prepare",
            )

    def _disability_weight_aggregator(self, pipeline):
        def _aggregate(group):
            return (pipeline(group.index) * to_years(group["step_size"])).sum()

        return _aggregate
