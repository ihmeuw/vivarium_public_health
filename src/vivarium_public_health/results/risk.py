"""
==============
Risk Observers
==============

This module contains tools for observing risk exposure during the simulation.

"""

from typing import Any, Dict, List, Optional

import pandas as pd
from vivarium.framework.engine import Builder

from vivarium_public_health.results.columns import COLUMNS
from vivarium_public_health.results.observer import PublicHealthObserver
from vivarium_public_health.utilities import to_years


class CategoricalRiskObserver(PublicHealthObserver):
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
            stratification:
                risk_name:
                    exclude:
                        - "sex"
                    include:
                        - "sample_stratification"
    """

    ##############
    # Properties #
    ##############

    @property
    def configuration_defaults(self) -> Dict[str, Any]:
        """
        A dictionary containing the defaults for any configurations managed by
        this component.
        """
        return {
            "stratification": {
                f"{self.risk}": super().configuration_defaults["stratification"][
                    self.get_configuration_name()
                ]
            }
        }

    @property
    def columns_required(self) -> Optional[List[str]]:
        return ["alive"]

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, risk: str) -> None:
        """
        Parameters
        ----------
        risk: name of a risk

        """
        super().__init__()
        self.risk = risk
        self.exposure_pipeline_name = f"{self.risk}.exposure"

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.step_size = builder.time.step_size()
        self.config = builder.configuration.stratification[self.risk]
        self.categories = builder.data.load(f"risk_factor.{self.risk}.categories")

    #################
    # Setup methods #
    #################

    def register_observations(self, builder: Builder) -> None:
        builder.results.register_stratification(
            f"{self.risk}",
            list(self.categories.keys()),
            requires_values=[self.exposure_pipeline_name],
        )
        self.register_adding_observation(
            builder=builder,
            name=f"person_time_{self.risk}",
            pop_filter=f'alive == "alive" and tracked==True',
            when="time_step__prepare",
            requires_columns=["alive"],
            requires_values=[self.exposure_pipeline_name],
            additional_stratifications=self.config.include + [self.risk],
            excluded_stratifications=self.config.exclude,
            aggregator=self.aggregate_risk_category_person_time,
        )

    ###############
    # Aggregators #
    ###############

    def aggregate_risk_category_person_time(self, x: pd.DataFrame) -> float:
        return len(x) * to_years(self.step_size())

    ##############################
    # Results formatting methods #
    ##############################

    def format(self, measure: str, results: pd.DataFrame) -> pd.DataFrame:
        results = results.reset_index()
        results.rename(columns={self.risk: COLUMNS.SUB_ENTITY}, inplace=True)
        return results

    def get_measure_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        return pd.Series("person_time", index=results.index)

    def get_entity_type_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        return pd.Series("rei", index=results.index)

    def get_entity_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        return pd.Series(self.risk, index=results.index)

    def get_sub_entity_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        # The sub-entity col was created in the 'format' method
        return results[COLUMNS.SUB_ENTITY]
