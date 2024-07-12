"""
==================
Mortality Observer
==================

This module contains tools for observing cause-specific and
excess mortality in the simulation, including "other causes".

"""

from typing import Any, Dict, List

import pandas as pd
from layered_config_tree import LayeredConfigTree
from vivarium.framework.engine import Builder

from vivarium_public_health.disease import DiseaseState, RiskAttributableDisease
from vivarium_public_health.results.columns import COLUMNS
from vivarium_public_health.results.observer import PublicHealthObserver


class MortalityObserver(PublicHealthObserver):
    """An observer for cause-specific deaths and ylls (including "other causes").

    By default, this counts cause-specific deaths and years of life lost over
    the full course of the simulation. It can be configured to add or remove
    stratification groups to the default groups defined by a
    :class:ResultsStratifier. The aggregate configuration key can be set to
    True to aggregate all deaths and ylls into a single observation and remove
    the stratification by cause of death to improve runtime.

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

    def __init__(self) -> None:
        super().__init__()
        self.required_death_columns = ["alive", "exit_time", "cause_of_death"]
        self.required_yll_columns = [
            "alive",
            "cause_of_death",
            "exit_time",
            "years_of_life_lost",
        ]

    ##############
    # Properties #
    ##############

    @property
    def mortality_classes(self) -> List:
        return [DiseaseState, RiskAttributableDisease]

    @property
    def configuration_defaults(self) -> Dict[str, Any]:
        """
        A dictionary containing the defaults for any configurations managed by
        this component.
        """
        config_defaults = super().configuration_defaults
        config_defaults["stratification"][self.get_configuration_name()]["aggregate"] = False
        return config_defaults

    @property
    def columns_required(self) -> List[str]:
        return [
            "alive",
            "years_of_life_lost",
            "cause_of_death",
            "exit_time",
        ]

    #################
    # Setup methods #
    #################

    def setup(self, builder: Builder) -> None:
        self.clock = builder.time.clock()
        self.causes_of_death = [
            cause
            for cause in builder.components.get_components_by_type(
                tuple(self.mortality_classes)
            )
            if cause.has_excess_mortality
        ]

    def get_configuration(self, builder: Builder) -> LayeredConfigTree:
        return builder.configuration.stratification[self.get_configuration_name()]

    def register_observations(self, builder: Builder) -> None:
        pop_filter = 'alive == "dead" and tracked == True'
        additional_stratifications = self.configuration.include
        if not self.configuration.aggregate:
            stratification_categories = [cause.state_id for cause in self.causes_of_death] + [
                "not_dead",
                "other_causes",
            ]
            builder.results.register_stratification(
                "cause_of_death",
                stratification_categories,
                requires_columns=["cause_of_death"],
            )
            additional_stratifications += ["cause_of_death"]
        self.register_adding_observation(
            builder=builder,
            name="deaths",
            pop_filter=pop_filter,
            requires_columns=self.required_death_columns,
            additional_stratifications=additional_stratifications,
            excluded_stratifications=self.configuration.exclude,
            aggregator=self.count_deaths,
        )
        self.register_adding_observation(
            builder=builder,
            name="ylls",
            pop_filter=pop_filter,
            requires_columns=self.required_yll_columns,
            additional_stratifications=additional_stratifications,
            excluded_stratifications=self.configuration.exclude,
            aggregator=self.calculate_ylls,
        )

    ###############
    # Aggregators #
    ###############

    def count_deaths(self, x: pd.DataFrame) -> float:
        died_of_cause = x["exit_time"] > self.clock()
        return sum(died_of_cause)

    def calculate_ylls(self, x: pd.DataFrame) -> float:
        died_of_cause = x["exit_time"] > self.clock()
        return x.loc[died_of_cause, "years_of_life_lost"].sum()

    ##############################
    # Results formatting methods #
    ##############################

    def format(self, measure: str, results: pd.DataFrame) -> pd.DataFrame:
        results = results.reset_index()
        if self.configuration.aggregate:
            results[COLUMNS.ENTITY] = "all_causes"
        else:
            results.rename(columns={"cause_of_death": COLUMNS.ENTITY}, inplace=True)
        return results[results[COLUMNS.ENTITY] != "not_dead"]

    def get_entity_type_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        entity_type_map = {cause.state_id: cause.cause_type for cause in self.causes_of_death}
        entity_type_map["other_causes"] = "cause"
        return results[COLUMNS.ENTITY].map(entity_type_map)

    def get_entity_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        # The entity col was created in the 'format' method
        return results[COLUMNS.ENTITY]

    def get_sub_entity_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        return results[COLUMNS.ENTITY]
