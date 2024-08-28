from typing import Callable, List, Optional, Union

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.results import Observer

from vivarium_public_health.results.columns import COLUMNS


class PublicHealthObserver(Observer):
    """A convenience class for typical public health observers.

    It exposes a method for registering the most common observation type
    (adding observation) as well methods for formatting public health results
    in a standardized way (to be overwritten as necessary).

    """

    def register_adding_observation(
        self,
        builder: Builder,
        name: str,
        pop_filter: str,
        when: str = "collect_metrics",
        requires_columns: List[str] = [],
        requires_values: List[str] = [],
        additional_stratifications: List[str] = [],
        excluded_stratifications: List[str] = [],
        aggregator_sources: Optional[List[str]] = None,
        aggregator: Callable[[pd.DataFrame], Union[float, pd.Series]] = len,
    ) -> None:
        """Registers an adding observation to the results system.

        An "adding" observation is one that adds/sums new results to existing
        result values. It is the most common type of observation used in public
        health models.

        Parameters
        ----------
        builder
            The builder object.
        name
            Name of the observation. It will also be the name of the output results
            file for this particular observation.
        pop_filter
            A Pandas query filter string to filter the population down to the
            simulants who should be considered for the observation.
        when
            Name of the lifecycle phase the observation should happen. Valid values are:
            "time_step__prepare", "time_step", "time_step__cleanup", or "collect_metrics".
        requires_columns
            List of the state table columns that are required by either the `pop_filter`
            or the `aggregator`.
        requires_values
            List of the value pipelines that are required by either the `pop_filter`
            or the `aggregator`.
        additional_stratifications
            List of additional stratification names by which to stratify this
            observation by.
        excluded_stratifications
            List of default stratification names to remove from this observation.
        aggregator_sources
            List of population view columns to be used in the `aggregator`.
        aggregator
            Function that computes the quantity for this observation.
        """
        builder.results.register_adding_observation(
            name=name,
            pop_filter=pop_filter,
            when=when,
            requires_columns=requires_columns,
            requires_values=requires_values,
            results_formatter=self.format_results,
            additional_stratifications=additional_stratifications,
            excluded_stratifications=excluded_stratifications,
            aggregator_sources=aggregator_sources,
            aggregator=aggregator,
        )

    def format_results(self, measure: str, results: pd.DataFrame) -> pd.DataFrame:
        """Top-level results formatter that calls standard sub-methods to be
        overwritten as necessary.

        Public health observations typically require four columns in addition to
        any stratifications and results columns: 'measure', 'entity_type', 'entity',
        and 'sub_entity'. This method provides a standardized way to format
        results by providing five sub-methods to be overwritten as necessary:
        - format()
        - get_measure_column()
        - get_entity_type_column()
        - get_entity_column()
        - get_sub_entity_column()

        Parameters
        ----------
        measure
            The measure name.
        results
            The raw results.

        Returns
        -------
            The formatted results.
        """

        results = self.format(measure, results)
        results[COLUMNS.MEASURE] = self.get_measure_column(measure, results)
        results[COLUMNS.ENTITY_TYPE] = self.get_entity_type_column(measure, results)
        results[COLUMNS.ENTITY] = self.get_entity_column(measure, results)
        results[COLUMNS.SUB_ENTITY] = self.get_sub_entity_column(measure, results)

        ordered_columns = [
            COLUMNS.MEASURE,
            COLUMNS.ENTITY_TYPE,
            COLUMNS.ENTITY,
            COLUMNS.SUB_ENTITY,
        ]
        ordered_columns += [
            c for c in results.columns if c not in ordered_columns + [COLUMNS.VALUE]
        ]
        ordered_columns += [COLUMNS.VALUE]
        return results[ordered_columns]

    def format(self, measure: str, results: pd.DataFrame) -> pd.DataFrame:
        """Format results.

        This method should be overwritten in subclasses to provide custom formatting
        for the results.

        Parameters
        ----------
        measure
            The measure name.
        results
            The raw results.

        Returns
        -------
            The formatted results.
        """
        return results

    def get_measure_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        """Get the 'measure' column.

        This method should be overwritten in subclasses to provide the 'measure' column.

        Parameters
        ----------
        measure
            The measure name.
        results
            The raw results.

        Returns
        -------
            The 'measure' column values.
        """
        return pd.Series(measure, index=results.index)

    def get_entity_type_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        """Get the 'entity_type' column.

        This method should be overwritten in subclasses to provide the 'entity_type' column.

        Parameters
        ----------
        measure
            The measure name.
        results
            The raw results.

        Returns
        -------
            The 'entity_type' column values.
        """
        return pd.Series(None, index=results.index)

    def get_entity_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        """Get the 'entity' column.

        This method should be overwritten in subclasses to provide the 'entity' column.

        Parameters
        ----------
        measure
            The measure name.
        results
            The raw results.

        Returns
        -------
            The 'entity' column values.
        """
        return pd.Series(None, index=results.index)

    def get_sub_entity_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        """Get the 'sub_entity' column.

        This method should be overwritten in subclasses to provide the 'sub_entity' column.

        Parameters
        ----------
        measure
            The measure name.
        results
            The raw results.

        Returns
        -------
            The 'sub_entity' column values.
        """
        return pd.Series(None, index=results.index)
