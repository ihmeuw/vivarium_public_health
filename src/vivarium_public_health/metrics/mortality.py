"""
==================
Mortality Observer
==================

This module contains tools for observing cause-specific and
excess mortality in the simulation, including "other causes".

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
    """An observer for cause-specific deaths and ylls (including "other causes").

    By default, this counts cause-specific deaths and years of life lost over
    the full course of the simulation. It can be configured to add or remove
    stratification groups to the default groups defined by a
    :class:ResultsStratifier.

    In the model specification, your configuration for this component should
    be specified as, e.g.:

    .. code-block:: yaml

        configuration:
            stratification:
                mortality:
                    exclude:
                        - "sex"
                    include:
                        - "sample_stratification"

    This observer needs to access the has_excess_mortality attribute of the causes
    we're observing, but this attribute gets defined in the setup of the cause models.
    As a result, the model specification should list this observer after causes.
    """

    configuration_defaults = {
        "stratification": {
            "mortality": {
                "exclude": [],
                "include": [],
            }
        }
    }

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
        self.clock = builder.time.clock()
        self.config = builder.configuration.stratification.mortality
        self._cause_components = builder.components.get_components_by_type(
            (DiseaseState, RiskAttributableDisease)
        )
        self.causes_of_death = ["other_causes"] + [
            cause.state_id for cause in self._cause_components if cause.has_excess_mortality
        ]

        columns_required = [
            "alive",
            "years_of_life_lost",
            "cause_of_death",
            "exit_time",
        ]

        self.population_view = builder.population.get_view(columns_required)

        for cause_of_death in self.causes_of_death:
            builder.results.register_observation(
                name=f"death_due_to_{cause_of_death}",
                pop_filter=f'alive == "dead" and cause_of_death == "{cause_of_death}"',
                aggregator=self.count_cause_specific_deaths,
                requires_columns=["alive", "cause_of_death", "exit_time"],
                additional_stratifications=self.config.include,
                excluded_stratifications=self.config.exclude,
                when="collect_metrics",
            )
            builder.results.register_observation(
                name=f"ylls_due_to_{cause_of_death}",
                pop_filter=f'alive == "dead" and cause_of_death == "{cause_of_death}"',
                aggregator=self.calculate_cause_specific_ylls,
                requires_columns=[
                    "alive",
                    "cause_of_death",
                    "exit_time",
                    "years_of_life_lost",
                ],
                additional_stratifications=self.config.include,
                excluded_stratifications=self.config.exclude,
                when="collect_metrics",
            )

    def count_cause_specific_deaths(self, x: pd.DataFrame) -> float:
        died_of_cause = x["exit_time"] > self.clock()
        return sum(died_of_cause)

    def calculate_cause_specific_ylls(self, x: pd.DataFrame) -> float:
        died_of_cause = x["exit_time"] > self.clock()
        return x.loc[died_of_cause, "years_of_life_lost"].sum()
