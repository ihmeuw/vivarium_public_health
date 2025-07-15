"""
======================
Intervention Observers
======================

This module contains tools for observing risk exposure during the simulation.

"""

import pandas as pd
from vivarium.framework.engine import Builder

from vivarium_public_health.results.columns import COLUMNS
from vivarium_public_health.results.observer import PublicHealthObserver
from vivarium_public_health.utilities import to_years


class CategoricalInterventionObserver(PublicHealthObserver):
    """
    A class for observering interventions. This class has the same implementation as
    the 'CategoricalRiskObserver' class.

    """

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, intervention: str) -> None:
        """Constructor for this observer.

        Parameters
        ----------
        intervention
            The name of the intervention being observed
        """
        super().__init__()
        self.intervention = intervention
        self.coverage_pipeline_name = f"{self.intervention}.coverage"

    #################
    # Setup methods #
    #################

    def setup(self, builder: Builder) -> None:
        """Set up the observer."""
        self.step_size = builder.time.step_size()
        self.categories = builder.data.load(f"intervention.{self.intervention}.categories")

    def get_configuration_name(self) -> str:
        return self.intervention

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
            f"{self.intervention}",
            list(self.categories.keys()),
            requires_values=[self.coverage_pipeline_name],
        )
        self.register_adding_observation(
            builder=builder,
            name=f"person_time_{self.intervention}",
            pop_filter=f'alive == "alive" and tracked==True',
            when="time_step__prepare",
            requires_columns=["alive"],
            requires_values=[self.coverage_pipeline_name],
            additional_stratifications=self.configuration.include + [self.intervention],
            excluded_stratifications=self.configuration.exclude,
            aggregator=self.aggregate_intervention_category_person_time,
        )

    ###############
    # Aggregators #
    ###############

    def aggregate_intervention_category_person_time(self, x: pd.DataFrame) -> float:
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
        results.rename(columns={self.intervention: COLUMNS.SUB_ENTITY}, inplace=True)
        return results

    def get_measure_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        """Get the 'measure' column values."""
        return pd.Series("person_time", index=results.index)

    def get_entity_type_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        """Get the 'entity_type' column values."""
        return pd.Series("rei", index=results.index)

    def get_entity_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        """Get the 'entity' column values."""
        return pd.Series(self.intervention, index=results.index)

    def get_sub_entity_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        """Get the 'sub_entity' column values."""
        # The sub-entity col was created in the 'format' method
        return results[COLUMNS.SUB_ENTITY]
