"""
==============
Risk Observers
==============

This module contains tools for observing risk exposure during the simulation.

"""
from collections import Counter
from typing import Dict, List

import pandas as pd
from vivarium.config_tree import ConfigTree
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import PopulationView
from vivarium.framework.values import Pipeline

from vivarium_public_health.metrics.stratification import ResultsStratifier
from vivarium_public_health.utilities import to_years


class CategoricalRiskObserver:
    """An observer for a categorical risk factor.

    Observes category person time for a risk factor.

    By default, this observer computes aggregate categorical person time
    over the full course of the simulation. It can be configured to add or
    remove stratification groups to the default groups defined by a
    ResultsStratifier.

    In the model specification, your configuration for this component should
    be specified as, e.g.:

    .. code-block:: yaml

        configuration:
            observers:
                risk_name:
                    exclude:
                        - "sex"
                    include:
                        - "sample_stratification"
    """

    configuration_defaults = {
        "observers": {
            "risk": {
                "exclude": [],
                "include": [],
            }
        }
    }

    def __init__(self, risk: str):
        """
        Parameters
        ----------
        risk :
        name of a risk

        """
        self.risk = risk
        self.configuration_defaults = self._get_configuration_defaults()

        self.exposure_pipeline_name = f"{self.risk}.exposure"
        self.metrics_pipeline_name = "metrics"

    def __repr__(self):
        return f"CategoricalRiskObserver({self.risk})"

    ##########################
    # Initialization methods #
    ##########################

    # noinspection PyMethodMayBeStatic
    def _get_configuration_defaults(self) -> Dict[str, Dict]:
        return {
            "observers": {
                f"{self.risk}": CategoricalRiskObserver.configuration_defaults["observers"][
                    "risk"
                ]
            }
        }

    ##############
    # Properties #
    ##############

    @property
    def name(self):
        return f"categorical_risk_observer.{self.risk}"

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder):
        self.config = self._get_stratification_configuration(builder)
        self.stratifier = self._get_stratifier(builder)
        self.categories = self._get_categories(builder)
        self.pipelines = self._get_pipelines(builder)
        self.population_view = self._get_population_view(builder)

        self.counts = Counter()

        self._register_time_step_prepare_listener(builder)
        self._register_metrics_modifier(builder)

    def _get_stratification_configuration(self, builder: Builder) -> ConfigTree:
        return builder.configuration.observers[self.risk]

    # noinspection PyMethodMayBeStatic
    def _get_stratifier(self, builder: Builder) -> ResultsStratifier:
        return builder.components.get_component(ResultsStratifier.name)

    def _get_categories(self, builder: Builder) -> List[str]:
        return builder.data.load(f"risk_factor.{self.risk}.categories")

    def _get_pipelines(self, builder: Builder) -> Dict[str, Pipeline]:
        return {
            self.exposure_pipeline_name: builder.value.get_value(self.exposure_pipeline_name)
        }

    # noinspection PyMethodMayBeStatic
    def _get_population_view(self, builder: Builder) -> PopulationView:
        columns_required = ["alive"]
        return builder.population.get_view(columns_required)

    def _register_time_step_prepare_listener(self, builder: Builder) -> None:
        # In order to get an accurate representation of person time we need to look at
        # the state table before anything happens.
        builder.event.register_listener("time_step__prepare", self.on_time_step_prepare)

    def _register_metrics_modifier(self, builder: Builder) -> None:
        builder.value.register_value_modifier(
            self.metrics_pipeline_name,
            modifier=self.metrics,
        )

    ########################
    # Event-driven methods #
    ########################

    def on_time_step_prepare(self, event: Event) -> None:
        step_size_in_years = to_years(event.step_size)
        pop = self.population_view.get(
            event.index, query="tracked == True and alive == 'alive'"
        )
        exposures = self.pipelines[self.exposure_pipeline_name](pop.index)
        groups = self.stratifier.group(
            exposures.index, self.config.include, self.config.exclude
        )
        for label, group_mask in groups:
            for category in self.categories:
                category_in_group_mask = group_mask & (exposures == category)
                person_time_in_group = category_in_group_mask.sum() * step_size_in_years
                new_observations = {
                    f"{self.risk}_{category}_person_time_{label}": person_time_in_group
                }
                self.counts.update(new_observations)

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    # noinspection PyUnusedLocal
    def metrics(self, index: pd.Index, metrics: Dict) -> Dict:
        metrics.update(self.counts)
        return metrics
