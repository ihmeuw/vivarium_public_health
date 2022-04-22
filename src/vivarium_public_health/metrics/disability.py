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
    with disability over the full course of the simulation. It can be
    configured to add or remove stratification groups to the default groups
    defined by a :class:ResultsStratifier.

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
        "observers": {
            "disability": {
                "exclude": [],
                "include": [],
            }
        }
    }

    def __init__(self):
        self.ylds_column_name = "years_lived_with_disability"
        self.metrics_pipeline_name = "metrics"
        self.disability_weight_pipeline_name = "disability_weight"

    def __repr__(self):
        return "DisabilityObserver()"

    ##############
    # Properties #
    ##############

    @property
    def name(self):
        return "disability_observer"

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.config = self._get_stratification_configuration(builder)
        self.stratifier = self._get_stratifier(builder)
        self.disability_weight = self._get_disability_weight_pipeline(builder)
        self._cause_components = self._get_cause_components(builder)
        self.disability_pipelines = {}
        self.population_view = self._get_population_view(builder)

        self.counts = Counter()

        self._register_post_setup_listener(builder)
        self._register_simulant_initializer(builder)
        self._register_time_step_prepare_listener(builder)
        self._register_metrics_modifier(builder)

    # noinspection PyMethodMayBeStatic
    def _get_stratification_configuration(self, builder: Builder) -> ConfigTree:
        return builder.configuration.observers.disability

    # noinspection PyMethodMayBeStatic
    def _get_stratifier(self, builder: Builder) -> ResultsStratifier:
        return builder.components.get_component(ResultsStratifier.name)

    def _get_disability_weight_pipeline(self, builder: Builder) -> Pipeline:
        # todo observer should not be creating the disability weight pipeline
        return builder.value.register_value_producer(
            self.disability_weight_pipeline_name,
            source=lambda index: [pd.Series(0.0, index=index)],
            preferred_combiner=list_combiner,
            preferred_post_processor=_disability_post_processor,
        )

    # noinspection PyMethodMayBeStatic
    def _get_cause_components(
        self, builder: Builder
    ) -> List[Union[DiseaseState, RiskAttributableDisease]]:
        return builder.components.get_components_by_type(
            (DiseaseState, RiskAttributableDisease)
        )

    # noinspection PyMethodMayBeStatic
    def _get_population_view(self, builder: Builder) -> PopulationView:
        columns_required = [
            self.ylds_column_name,
        ]
        return builder.population.get_view(columns_required)

    def _register_post_setup_listener(self, builder: Builder) -> None:
        builder.event.register_listener("post_setup", self.on_post_setup)

    def _register_simulant_initializer(self, builder: Builder) -> None:
        # todo observer should not be modifying state table
        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            creates_columns=[self.ylds_column_name],
        )

    def _register_time_step_prepare_listener(self, builder: Builder) -> None:
        # In order to get an accurate representation of person time we need to look at
        # the state table before anything happens.
        builder.event.register_listener("time_step__prepare", self.on_time_step_prepare)

    def _register_metrics_modifier(self, builder: Builder) -> None:
        builder.value.register_value_modifier(
            self.metrics_pipeline_name,
            modifier=self.metrics,
            requires_columns=[self.ylds_column_name],
        )

    ########################
    # Event-driven methods #
    ########################

    # noinspection PyUnusedLocal
    def on_post_setup(self, event: Event) -> None:
        for cause in self._cause_components:
            if cause.has_disability:
                self.disability_pipelines[cause.state_id] = cause.disability_weight

    def on_initialize_simulants(self, pop_data: pd.DataFrame) -> None:
        self.population_view.update(
            pd.Series(0.0, index=pop_data.index, name=self.ylds_column_name)
        )

    def on_time_step_prepare(self, event: Event) -> None:
        step_size_in_years = to_years(event.step_size)
        pop = self.population_view.get(
            event.index, query='tracked == True and alive == "alive"'
        )
        groups = self.stratifier.group(pop.index, self.config.include, self.config.exclude)
        for label, group_mask in groups:
            group_index = pop[group_mask].index
            for cause, disability_weight in self.disability_pipelines.items():
                disability_weight = disability_weight(group_index).sum() * step_size_in_years
                new_observations = {
                    f"ylds_due_to_{cause}_{label}": disability_weight,
                }
                self.counts.update(new_observations)

        pop.loc[:, self.ylds_column_name] += self.disability_weight(pop.index)
        self.population_view.update(pop)

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def metrics(self, index: pd.Index, metrics: Dict) -> Dict:
        total_ylds = self.population_view.get(index)[self.ylds_column_name].sum()
        metrics["years_lived_with_disability"] = total_ylds
        metrics.update(self.counts)
        return metrics


def _disability_post_processor(value: NumberLike, step_size: pd.Timedelta) -> NumberLike:
    return rescale_post_processor(union_post_processor(value, step_size), step_size)
