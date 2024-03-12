"""
==================
Mortality Observer
==================

This module contains tools for observing cause-specific and
excess mortality in the simulation, including "other causes".

"""
from typing import Callable, List, Optional

import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder

from vivarium_public_health.disease import DiseaseState, RiskAttributableDisease


class MortalityObserver(Component):
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

    CONFIGURATION_DEFAULTS = {
        "stratification": {
            "mortality": {
                "exclude": [],
                "include": [],
                "aggregate": False,
            }
        }
    }

    ##############
    # Properties #
    ##############

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
        self._cause_components = builder.components.get_components_by_type(
            (DiseaseState, RiskAttributableDisease)
        )
        self.causes_of_death = ["other_causes"] + [
            cause.state_id for cause in self._cause_components if cause.has_excess_mortality
        ]
        self.required_death_columns = ["alive", "exit_time"]
        self.required_yll_columns = [
            "alive",
            "cause_of_death",
            "exit_time",
            "years_of_life_lost",
        ]
        if not self.config.aggregate:
            for cause_of_death in self.causes_of_death:
                self._register_mortality_observations(
                    builder, cause_of_death, f'cause_of_death == "{cause_of_death}"'
                )
        else:
            self._register_mortality_observations(builder, "all_causes")

    ###################
    # Private methods #
    ###################

    def _register_mortality_observations(
        self, builder: Builder, cause: str, additional_pop_filter: str = ""
    ) -> None:
        pop_filter = (
            'alive == "dead" and tracked == True'
            if additional_pop_filter == ""
            else f'alive == "dead" and tracked == True and {additional_pop_filter}'
        )
        builder.results.register_observation(
            name=f"death_due_to_{cause}",
            pop_filter=pop_filter,
            aggregator=self.count_deaths,
            requires_columns=self.required_death_columns,
            additional_stratifications=self.config.include,
            excluded_stratifications=self.config.exclude,
            when="collect_metrics",
        )
        builder.results.register_observation(
            name=f"ylls_due_to_{cause}",
            pop_filter=pop_filter,
            aggregator=self.calculate_ylls,
            requires_columns=self.required_yll_columns,
            additional_stratifications=self.config.include,
            excluded_stratifications=self.config.exclude,
            when="collect_metrics",
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
