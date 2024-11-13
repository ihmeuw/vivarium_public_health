"""
==============
Risk Observers
==============

This module contains tools for observing risk exposure during the simulation.

"""

from typing import Any

import pandas as pd
from layered_config_tree import LayeredConfigTree
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

    Attributes
    ----------
    risk
        The name of the risk factor.
    exposure_pipeline_name
        The name of the pipeline that produces the risk factor exposure.
    step_size
        The time step size of the simulation.
    categories
        The categories of the risk factor.

    """

    ##############
    # Properties #
    ##############

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        """A dictionary containing the defaults for any configurations managed by
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
    def columns_required(self) -> list[str] | None:
        """The columns required by this observer."""
        return ["alive"]

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, risk: str) -> None:
        """Constructor for this observer.

        Parameters
        ----------
        risk
            The name of the risk being observed
        """
        super().__init__()
        self.risk = risk
        self.exposure_pipeline_name = f"{self.risk}.exposure"

    #################
    # Setup methods #
    #################

    def setup(self, builder: Builder) -> None:
        """Set up the observer."""
        self.step_size = builder.time.step_size()
        self.categories = builder.data.load(f"risk_factor.{self.risk}.categories")

    def get_configuration(self, builder: Builder) -> LayeredConfigTree:
        """Get the stratification configuration for this observer.

        Parameters
        ----------
        builder
            The builder object for the simulation.

        Returns
        -------
            The stratification configuration for this observer.
        """
        return builder.configuration.stratification[self.risk]

    def register_observations(self, builder: Builder) -> None:
        """Register a stratification and observation.

        Notes
        -----
        While it's typical for all stratification registrations to be encapsulated
        in a single class (i.e. the
        :class:ResultsStratifier <vivarium_public_health.results.stratification.ResultsStratifier),
        this observer registers an additional one. While it could be registered
        in the ``ResultsStratifier`` as well, it is specific to this observer and
        so it is registered here while we have easy access to the required categories
        and value names.
        """
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
            additional_stratifications=self.configuration.include + [self.risk],
            excluded_stratifications=self.configuration.exclude,
            aggregator=self.aggregate_risk_category_person_time,
        )

    ###############
    # Aggregators #
    ###############

    def aggregate_risk_category_person_time(self, x: pd.DataFrame) -> float:
        """Aggregate the person time for this time step."""
        return len(x) * to_years(self.step_size())

    ##############################
    # Results formatting methods #
    ##############################

    def format(self, measure: str, results: pd.DataFrame) -> pd.DataFrame:
        """Rename the appropriate column to 'sub_entity'.

        The primary thing this method does is rename the risk column
        to 'sub_entity'. We do this here instead of the 'get_sub_entity_column'
        method simply because we do not want the risk column at all. If we keep
        it here and then return it as the sub-entity column later, the final
        results would have both.

        Parameters
        ----------
        measure
            The measure.
        results
            The results to format.

        Returns
        -------
            The formatted results.
        """
        results = results.reset_index()
        results.rename(columns={self.risk: COLUMNS.SUB_ENTITY}, inplace=True)
        return results

    def get_measure_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        """Get the 'measure' column values."""
        return pd.Series("person_time", index=results.index)

    def get_entity_type_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        """Get the 'entity_type' column values."""
        return pd.Series("rei", index=results.index)

    def get_entity_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        """Get the 'entity' column values."""
        return pd.Series(self.risk, index=results.index)

    def get_sub_entity_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        """Get the 'sub_entity' column values."""
        # The sub-entity col was created in the 'format' method
        return results[COLUMNS.SUB_ENTITY]
