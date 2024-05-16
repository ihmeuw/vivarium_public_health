"""
==================
Mortality Observer
==================

This module contains tools for observing cause-specific and
excess mortality in the simulation, including "other causes".

"""

from functools import partial
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.results import StratifiedObserver

from vivarium_public_health.disease import DiseaseState, RiskAttributableDisease
from vivarium_public_health.metrics.reporters import write_dataframe_to_parquet


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

    def __init__(self):
        super().__init__()
        self.required_death_columns = ["alive", "exit_time"]
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

    def register_observations(self, builder: Builder) -> None:
        disease_components = builder.components.get_components_by_type(
            tuple(self.mortality_classes)
        )
        if not self.config.aggregate:
            causes_of_death = [
                cause for cause in disease_components if cause.has_excess_mortality
            ]
            for cause_of_death in causes_of_death:
                self._register_mortality_observations(
                    builder, cause_of_death, f'cause_of_death == "{cause_of_death.state_id}"'
                )
            self._register_mortality_observations(
                builder, "other_causes", 'cause_of_death == "other_causes"'
            )
        else:
            self._register_mortality_observations(builder, "all_causes")

    ###################
    # Private methods #
    ###################

    def _register_mortality_observations(
        self,
        builder: Builder,
        cause_state: Union[str, DiseaseState, RiskAttributableDisease],
        additional_pop_filter: Optional[str] = None,
    ) -> None:
        basic_filter = 'alive == "dead" and tracked == True'
        pop_filter = (
            basic_filter
            if not additional_pop_filter
            else " and ".join([basic_filter, additional_pop_filter])
        )
        measure = cause_state.state_id if not isinstance(cause_state, str) else cause_state
        builder.results.register_observation(
            name=f"deaths_due_to_{measure}",
            pop_filter=pop_filter,
            aggregator=self.count_deaths,
            requires_columns=self.required_death_columns,
            additional_stratifications=self.config.include,
            excluded_stratifications=self.config.exclude,
            when="collect_metrics",
            report=partial(self.write_mortality_results, cause_state),
        )
        builder.results.register_observation(
            name=f"ylls_due_to_{measure}",
            pop_filter=pop_filter,
            aggregator=self.calculate_ylls,
            requires_columns=self.required_yll_columns,
            additional_stratifications=self.config.include,
            excluded_stratifications=self.config.exclude,
            when="collect_metrics",
            report=partial(self.write_mortality_results, cause_state),
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
        cause_state: Union[str, DiseaseState, RiskAttributableDisease],
        measure: str,
        results: pd.DataFrame,
    ) -> None:
        measure_name = measure.split("_due_to_")[0]
        kwargs = {
            "entity_type": (
                cause_state.cause_type if not isinstance(cause_state, str) else "cause"
            ),
            "entity": cause_state.model if not isinstance(cause_state, str) else cause_state,
            "sub_entity": cause_state.state_id if not isinstance(cause_state, str) else None,
            "results_dir": self.results_dir,
            "random_seed": self.random_seed,
            "input_draw": self.input_draw,
        }
        write_dataframe_to_parquet(
            results=results,
            measure=measure_name,
            **kwargs,
        )
