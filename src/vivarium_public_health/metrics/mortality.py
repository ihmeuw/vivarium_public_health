"""
==================
Mortality Observer
==================

This module contains tools for observing cause-specific and
excess mortality in the simulation, including "other causes".

"""

from typing import Any, Dict, List, Optional

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.results import StratifiedObserver

from vivarium_public_health.disease import DiseaseState, RiskAttributableDisease
from vivarium_public_health.metrics.reporters import COLUMNS, write_dataframe


class MortalityObserver(StratifiedObserver):
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
        config_defaults["stratification"]["mortality"]["aggregate"] = False
        return config_defaults

    @property
    def columns_required(self) -> Optional[List[str]]:
        return [
            "alive",
            "years_of_life_lost",
            "cause_of_death",
            "exit_time",
        ]

    #####################
    # Lifecycle methods #
    #####################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.clock = builder.time.clock()
        self.config = builder.configuration.stratification.mortality
        self.causes_of_death = [
            cause
            for cause in builder.components.get_components_by_type(
                tuple(self.mortality_classes)
            )
            if cause.has_excess_mortality
        ]

    def register_observations(self, builder: Builder) -> None:
        pop_filter = 'alive == "dead" and tracked == True'
        additional_stratifications = self.config.include
        if not self.config.aggregate:
            additional_stratifications += ["cause_of_death"]
            builder.results.register_stratification(
                "cause_of_death",
                [cause.state_id for cause in self.causes_of_death]
                + ["not_dead", "other_causes"],
                requires_columns=["cause_of_death"],
            )
        builder.results.register_observation(
            name="deaths",
            pop_filter=pop_filter,
            aggregator=self.count_deaths,
            requires_columns=self.required_death_columns,
            additional_stratifications=additional_stratifications,
            excluded_stratifications=self.config.exclude,
            when="collect_metrics",
            report=self.write_mortality_results,
        )
        builder.results.register_observation(
            name="ylls",
            pop_filter=pop_filter,
            aggregator=self.calculate_ylls,
            requires_columns=self.required_yll_columns,
            additional_stratifications=additional_stratifications,
            excluded_stratifications=self.config.exclude,
            when="collect_metrics",
            report=self.write_mortality_results,
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

    ##################
    # Report methods #
    ##################

    def write_mortality_results(
        self,
        measure: str,
        results: pd.DataFrame,
    ) -> None:
        """Format dataframe and write out"""

        results = results.reset_index()

        if "cause_of_death" in results.columns:
            results.rename(columns={"cause_of_death": COLUMNS.ENTITY}, inplace=True)
        else:
            # self.aggregate_causes is True
            results[COLUMNS.ENTITY] = "all_causes"

        results = results[results[COLUMNS.ENTITY] != "not_dead"]
        results[COLUMNS.MEASURE] = measure

        results.loc[results[COLUMNS.ENTITY] == "other_causes", COLUMNS.ENTITY_TYPE] = "cause"
        results.loc[
            results[COLUMNS.ENTITY] == "other_causes", COLUMNS.SUB_ENTITY
        ] = "other_causes"

        results.loc[results[COLUMNS.ENTITY] == "all_causes", COLUMNS.ENTITY_TYPE] = "cause"
        results.loc[
            results[COLUMNS.ENTITY] == "all_causes", COLUMNS.SUB_ENTITY
        ] = "all_causes"

        for cause in self.causes_of_death:
            results.loc[
                results[COLUMNS.ENTITY] == cause.state_id, COLUMNS.ENTITY_TYPE
            ] = cause.cause_type
            results.loc[
                results[COLUMNS.ENTITY] == cause.state_id, COLUMNS.SUB_ENTITY
            ] = cause.state_id

        results["random_seed"] = self.random_seed
        results["input_draw"] = self.input_draw

        # Reorder columns so stratifcations are first and value is last
        results = results[
            [c for c in results.columns if c != COLUMNS.VALUE] + [COLUMNS.VALUE]
        ]

        write_dataframe(
            results=results,
            measure=measure,
            results_dir=self.results_dir,
        )
