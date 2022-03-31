"""
===================
Disability Observer
===================

This module contains tools for observing years lived with disability (YLDs)
in the simulation.

"""
from collections import Counter
from typing import Dict

import pandas as pd
from vivarium.config_tree import ConfigTree
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import PopulationView
from vivarium.framework.time import Timedelta
from vivarium.framework.values import (
    Pipeline,
    list_combiner,
    rescale_post_processor,
    union_post_processor,
)

from vivarium_public_health.disease import DiseaseState, RiskAttributableDisease
from vivarium_public_health.metrics.stratification import ResultsStratifier
from vivarium_public_health.utilities import to_time_delta, to_years


class DisabilityObserver:
    """Counts years lived with disability.

    By default, this counts both aggregate and cause-specific years lived
    with disability over the full course of the simulation. It can be
    configured to bin the cause-specific YLDs into age groups, sexes, and years
    by setting the ``by_age``, ``by_sex``, and ``by_year`` flags, respectively.

    In the model specification, your configuration for this component should
    be specified as, e.g.:

    .. code-block:: yaml

        configuration:
            metrics:
                disability:
                    by_age: True
                    by_year: False
                    by_sex: True

    """

    configuration_defaults = {
        "include": [],
        "exclude": [],
    }

    def __init__(self):
        self.configuration_defaults = self._get_configuration_defaults()

        self.ylds_column_name = "years_lived_with_disability"
        self.metrics_pipeline_name = "metrics"
        self.disability_weight_pipeline_name = "disability_weight"

    def __repr__(self):
        return "DisabilityObserver()"

    ##########################
    # Initialization methods #
    ##########################

    # noinspection PyMethodMayBeStatic
    def _get_configuration_defaults(self) -> Dict[str, Dict]:
        return {
            "observers": {
                "disability": DisabilityObserver.configuration_defaults
            }
        }

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
        self.step_size = self._get_step_size(builder)
        self.stratifier = self._get_stratifier(builder)
        self.disability_weight = self._get_disability_weight_pipeline(builder)
        self.causes_of_disability_pipelines = self._get_causes_of_disability_pipelines(builder)
        self.population_view = self._get_population_view(builder)

        self.counts = Counter()

        self._register_simulant_initializer(builder)
        self._register_time_step_prepare_listener(builder)
        self._register_metrics_modifier(builder)

    # noinspection PyMethodMayBeStatic
    def _get_stratification_configuration(self, builder: Builder) -> ConfigTree:
        return builder.configuration.observers.disability

    # noinspection PyMethodMayBeStatic
    def _get_step_size(self, builder: Builder) -> Timedelta:
        return to_time_delta(builder.configuration.time.step_size)

    # noinspection PyMethodMayBeStatic
    def _get_stratifier(self, builder: Builder) -> ResultsStratifier:
        return builder.components.get_component(ResultsStratifier.NAME)

    def _get_disability_weight_pipeline(self, builder: Builder) -> Pipeline:
        # todo observer should not be creating the disability weight pipeline
        return builder.value.register_value_producer(
            self.disability_weight_pipeline_name,
            source=lambda index: [pd.Series(0.0, index=index)],
            preferred_combiner=list_combiner,
            preferred_post_processor=_disability_post_processor,
        )

    # noinspection PyMethodMayBeStatic
    def _get_causes_of_disability_pipelines(self, builder: Builder) -> Dict[str, Pipeline]:
        # todo can we specify only causes with a disability weight?
        causes_of_disability = builder.components.get_components_by_type(
            (DiseaseState, RiskAttributableDisease)
        )
        return {
            cause.state_id: builder.value.get_value(f"{cause.state_id}.disability_weight")
            for cause in causes_of_disability
        }

    # noinspection PyMethodMayBeStatic
    def _get_population_view(self, builder: Builder) -> PopulationView:
        columns_required = [
            self.ylds_column_name,
        ]
        return builder.population.get_view(columns_required)

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

    def on_initialize_simulants(self, pop_data: pd.DataFrame) -> None:
        self.population_view.update(
            pd.Series(0.0, index=pop_data.index, name=self.ylds_column_name)
        )

    def on_time_step_prepare(self, event: Event) -> None:
        pop = self.population_view.get(
            event.index, query='tracked == True and alive == "alive"'
        )
        groups = self.stratifier.group(
            pop.index, set(self.config.include), set(self.config.exclude)
        )
        for label, group_index in groups:
            for cause, disability_weight in self.causes_of_disability_pipelines.items():
                new_observations = {
                    f"ylds_due_to_{cause}_{label}":
                        disability_weight(group_index).sum() * to_years(self.step_size)
                }
                self.counts.update(new_observations)

        pop.loc[:, self.ylds_column_name] += self.disability_weight(pop.index)
        self.population_view.update(pop)

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def metrics(self, index: pd.Index, metrics: Dict):
        total_ylds = self.population_view.get(index)[self.ylds_column_name].sum()
        metrics["years_lived_with_disability"] = total_ylds
        metrics.update(self.counts)
        return metrics


def _disability_post_processor(value, step_size):
    return rescale_post_processor(union_post_processor(value, step_size), step_size)
