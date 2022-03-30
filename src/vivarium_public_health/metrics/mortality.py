"""
==================
Mortality Observer
==================

This module contains tools for observing all-cause, cause-specific, and
excess mortality in the simulation.

"""
from collections import Counter
from typing import Dict, Set

import pandas as pd

from vivarium.framework.engine import Builder, ConfigTree
from vivarium.framework.event import Event
from vivarium.framework.population import PopulationView
from vivarium.framework.time import Timedelta

from vivarium_public_health.disease import DiseaseState, RiskAttributableDisease
from vivarium_public_health.metrics.stratification import ResultsStratifier
from vivarium_public_health.utilities import to_time_delta


class MortalityObserver:
    """An observer for cause-specific deaths, ylls, and total person time.

    By default, this counts cause-specific deaths and years of life lost over
    the full course of the simulation. It can be configured to add or remove
    stratification groups to the default groups defined by a ResultsStratifier.

    In the model specification, your configuration for this component should
    be specified as, e.g.:

    .. code-block:: yaml

        configuration:
            observers:
                mortality:
                    exclude:
                        - "year"
                    include:
                        - "death_year"
                        - "cause_of_death"



    """

    configuration_defaults = {
        "exclude": [],
        "include": [],
    }

    def __init__(self):
        self.configuration_defaults = self._get_configuration_defaults()

        self.metrics_pipeline_name = "metrics"
        self.tmrle_key = "population.theoretical_minimum_risk_life_expectancy"

    ##########################
    # Initialization methods #
    ##########################

    # noinspection PyMethodMayBeStatic
    def _get_configuration_defaults(self) -> Dict[str, Dict]:
        return {
            "observers": {
                "mortality": MortalityObserver.configuration_defaults
            }
        }

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
        self.time_step = self._get_time_step(builder)
        self.counts = Counter()
        self.stratifier = self._get_stratifier(builder)
        self.population_view = self._get_population_view(builder)
        self.causes_of_death = self._get_causes_of_death(builder)

        self.register_collect_metrics_listener(builder)
        self.register_metrics_modifier(builder)

    # noinspection PyMethodMayBeStatic
    def _get_stratification_configuration(self, builder: Builder) -> ConfigTree:
        return builder.configuration.observers.mortality

    # noinspection PyMethodMayBeStatic
    def _get_time_step(self, builder: Builder) -> Timedelta:
        return to_time_delta(builder.configuration.time.step_size)

    # noinspection PyMethodMayBeStatic
    def _get_stratifier(self, builder: Builder) -> ResultsStratifier:
        return builder.components.get_component(ResultsStratifier.NAME)

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

    # noinspection PyMethodMayBeStatic
    def _get_causes_of_death(self, builder: Builder) -> Set[str]:
        # todo can we specify only causes with excess mortality?
        diseases = builder.components.get_components_by_type(
            (DiseaseState, RiskAttributableDisease)
        )
        return {c.state_id for c in diseases} | {"other_causes"}

    def register_collect_metrics_listener(self, builder: Builder) -> None:
        builder.event.register_listener("time_step", self.on_collect_metrics)

    def register_metrics_modifier(self, builder: Builder) -> None:
        builder.value.register_value_modifier(
            self.metrics_pipeline_name,
            modifier=self.metrics,
            requires_columns=["age", "exit_time", "alive"],
        )

    ########################
    # Event-driven methods #
    ########################

    def on_collect_metrics(self, event: Event) -> None:
        pop = self.population_view.get(event.index)
        pop_died = pop[(pop["alive"] == "dead") & (pop["exit_time"] > event.time - self.time_step)]

        groups = self.stratifier.group(
            pop_died.index, set(self.config.include), set(self.config.exclude)
        )
        for label, group_index in groups:
            for cause in self.causes_of_death:
                pop_died_of_cause = (
                    pop_died.loc[group_index, :]
                    .query(f"cause_of_death == '{cause}'")
                )
                new_observations = {
                    f"death_due_to_{cause}_{label}": pop_died_of_cause.size,
                    f"ylls_due_to_{cause}_{label}": pop_died_of_cause["years_of_life_lost"].sum()
                }
                self.counts.update(new_observations)

    def metrics(self, index: pd.Index, metrics: Dict) -> Dict:
        pop = self.population_view.get(index)

        the_living = pop[(pop.alive == "alive") & pop.tracked]
        the_dead = pop[pop.alive == "dead"]
        metrics["years_of_life_lost"] = the_dead["years_of_life_lost"].sum()
        metrics["total_population_living"] = the_living.size
        metrics["total_population_dead"] = the_dead.size
        metrics.update(self.counts)

        return metrics

    def __repr__(self):
        return "MortalityObserver()"
