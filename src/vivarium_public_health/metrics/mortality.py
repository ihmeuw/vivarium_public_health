"""
==================
Mortality Observer
==================

This module contains tools for observing all-cause, cause-specific, and
excess mortality in the simulation.

"""
from typing import Dict, List

import pandas as pd

from vivarium.framework.engine import Builder, ConfigTree
from vivarium.framework.population import PopulationView

from vivarium_public_health.metrics.stratification import ResultsStratifier


class MortalityObserver:
    """An observer for cause-specific deaths, ylls, and total person time.

    By default, this counts cause-specific deaths, years of life lost, and
    total person time over the full course of the simulation. It can be
    configured to bin these measures into age groups, sexes, and years
    by setting the ``by_age``, ``by_sex``, and ``by_year`` flags, respectively.

    In the model specification, your configuration for this component should
    be specified as, e.g.:

    .. code-block:: yaml

        configuration:
            metrics:
                mortality:
                    by_age: True
                    by_year: False
                    by_sex: True

    """

    configuration_defaults = {
        "include": [ResultsStratifier.DEATH_YEAR, ResultsStratifier.CAUSE_OF_DEATH],
        "exclude": [ResultsStratifier.YEAR],
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
            "metrics": {
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
        self.stratifier = self._get_stratifier(builder)
        self.population_view = self._get_population_view(builder)

        self.register_metrics_modifier(builder)

    # noinspection PyMethodMayBeStatic
    def _get_default_stratifications(self, builder: Builder) -> List[str]:
        return builder.configuration.metrics.default

    # noinspection PyMethodMayBeStatic
    def _get_stratification_configuration(self, builder: Builder) -> ConfigTree:
        return builder.configuration.metrics.mortality

    # noinspection PyMethodMayBeStatic
    def _get_stratifier(self, builder: Builder) -> ResultsStratifier:
        return builder.components.get_component(ResultsStratifier.NAME)

    # noinspection PyMethodMayBeStatic
    def _get_population_view(self, builder: Builder) -> PopulationView:
        columns_required = [
            "tracked",
            "alive",
            "years_of_life_lost",
            "age",
        ]
        return builder.population.get_view(columns_required)

    def register_metrics_modifier(self, builder: Builder) -> None:
        builder.value.register_value_modifier(
            self.metrics_pipeline_name,
            modifier=self.metrics,
            requires_columns=["age", "exit_time", "alive"],
        )

    ########################
    # Event-driven methods #
    ########################

    def metrics(self, index: pd.Index, metrics: Dict) -> Dict:
        pop = self.population_view.get(index)

        groups = self.stratifier.group(pop, set(self.config.include), set(self.config.exclude))
        for label, pop_in_group in groups:
            new_observations = {
                f"death_{label}": pop_in_group.size,
                f"ylls_{label}": pop_in_group["years_of_life_lost"].sum()
            }
            metrics.update(new_observations)

        the_living = pop[(pop.alive == "alive") & pop.tracked]
        the_dead = pop[pop.alive == "dead"]
        metrics["years_of_life_lost"] = the_dead["years_of_life_lost"].sum()
        metrics["total_population_living"] = the_living.size
        metrics["total_population_dead"] = the_dead.size

        return metrics

    def __repr__(self):
        return "MortalityObserver()"
