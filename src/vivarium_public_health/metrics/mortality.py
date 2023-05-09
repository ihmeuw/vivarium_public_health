"""
==================
Mortality Observer
==================

This module contains tools for observing all-cause, cause-specific, and
excess mortality in the simulation.

"""
from collections import Counter
from typing import Dict, List, Union

import pandas as pd
from vivarium.framework.engine import Builder, ConfigTree
from vivarium.framework.event import Event
from vivarium.framework.population import PopulationView

from vivarium_public_health.disease import DiseaseState, RiskAttributableDisease
from vivarium_public_health.metrics.stratification import ResultsStratifier


class MortalityObserver:
    """An observer for cause-specific deaths, ylls, and total person time.

    By default, this counts cause-specific deaths and years of life lost over
    the full course of the simulation. It can be configured to add or remove
    stratification groups to the default groups defined by a
    :class:ResultsStratifier.

    In the model specification, your configuration for this component should
    be specified as, e.g.:

    .. code-block:: yaml

        configuration:
            observers:
                mortality:
                    exclude:
                        - "sex"
                    include:
                        - "sample_stratification"
    """

    configuration_defaults = {
        "observers": {
            "mortality": {
                "exclude": [],
                "include": [],
            }
        }
    }

    def __init__(self):
        self.metrics_pipeline_name = "metrics"
        self.tmrle_key = "population.theoretical_minimum_risk_life_expectancy"

    def __repr__(self):
        return "MortalityObserver()"

    ##############
    # Properties #
    ##############

    @property
    def name(self):
        return "mortality_observer"

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder):
        self.config = self._get_stratification_configuration(builder)
        self.stratifier = self._get_stratifier(builder)
        self._cause_components = self._get_cause_components(builder)
        self.causes_of_death = ["other_causes"]
        self.population_view = self._get_population_view(builder)

        self.counts = Counter()

        self._register_post_setup_listener(builder)
        self._register_collect_metrics_listener(builder)
        self._register_metrics_modifier(builder)

    # noinspection PyMethodMayBeStatic
    def _get_stratification_configuration(self, builder: Builder) -> ConfigTree:
        return builder.configuration.observers.mortality

    # noinspection PyMethodMayBeStatic
    def _get_stratifier(self, builder: Builder) -> ResultsStratifier:
        return builder.components.get_component(ResultsStratifier.name)

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
            "tracked",
            "alive",
            "years_of_life_lost",
            "cause_of_death",
            "exit_time",
        ]
        return builder.population.get_view(columns_required)

    def _register_post_setup_listener(self, builder: Builder) -> None:
        builder.event.register_listener("post_setup", self.on_post_setup)

    def _register_collect_metrics_listener(self, builder: Builder) -> None:
        builder.event.register_listener("time_step", self.on_collect_metrics)

    def _register_metrics_modifier(self, builder: Builder) -> None:
        builder.value.register_value_modifier(
            self.metrics_pipeline_name,
            modifier=self.metrics,
            requires_columns=["age", "exit_time", "alive"],
        )

    ########################
    # Event-driven methods #
    ########################

    # noinspection PyUnusedLocal
    def on_post_setup(self, event: Event) -> None:
        self.causes_of_death += [
            cause.state_id for cause in self._cause_components if cause.has_excess_mortality
        ]

    def on_collect_metrics(self, event: Event) -> None:
        pop = self.population_view.get(event.index)
        pop_died = pop[(pop["alive"] == "dead") & (pop["exit_time"] == event.time)]

        groups = self.stratifier.group(
            pop_died.index, self.config.include, self.config.exclude
        )
        for label, group_mask in groups:
            for cause in self.causes_of_death:
                cause_mask = pop_died["cause_of_death"] == cause
                pop_died_of_cause = pop_died[group_mask & cause_mask]
                new_observations = {
                    f"death_due_to_{cause}_{label}": pop_died_of_cause.index.size,
                    f"ylls_due_to_{cause}_{label}": pop_died_of_cause[
                        "years_of_life_lost"
                    ].sum(),
                }
                self.counts.update(new_observations)

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def metrics(self, index: pd.Index, metrics: Dict) -> Dict:
        pop = self.population_view.get(index)

        the_living = pop[(pop.alive == "alive") & pop.tracked]
        the_dead = pop[pop.alive == "dead"]
        metrics["years_of_life_lost"] = the_dead["years_of_life_lost"].sum()
        metrics["total_population_living"] = len(the_living)
        metrics["total_population_dead"] = len(the_dead)
        metrics.update(self.counts)

        return metrics
