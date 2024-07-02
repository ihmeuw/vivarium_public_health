from typing import Callable, List, Optional, Union

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.results import Observer

from vivarium_public_health.results.columns import COLUMNS


class PublicHealthObserver(Observer):
    """A convenience class for typical public health observers. It provides
    an entry point for registering the most common observation type
    as well as standardized results formatting methods to overwrite as necessary.
    """

    def register_adding_observation(
        self,
        builder: Builder,
        name,
        pop_filter,
        when: str = "collect_metrics",
        requires_columns: List[str] = [],
        requires_values: List[str] = [],
        additional_stratifications: List[str] = [],
        excluded_stratifications: List[str] = [],
        aggregator_sources: Optional[List[str]] = None,
        aggregator: Callable[[pd.DataFrame], Union[float, pd.Series]] = len,
    ):
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
        """

        results = self.format(measure, results)
        results[COLUMNS.MEASURE] = self.get_measure_col(measure, results)
        results[COLUMNS.ENTITY_TYPE] = self.get_entity_type_col(measure, results)
        results[COLUMNS.ENTITY] = self.get_entity_col(measure, results)
        results[COLUMNS.SUB_ENTITY] = self.get_sub_entity_col(measure, results)

        return results[[c for c in results.columns if c != COLUMNS.VALUE] + [COLUMNS.VALUE]]

    def format(self, measure: str, results: pd.DataFrame) -> pd.DataFrame:
        return results

    def get_measure_col(self, measure: str, results: pd.DataFrame) -> pd.Series:
        return pd.Series(measure, index=results.index)

    def get_entity_type_col(self, measure: str, results: pd.DataFrame) -> pd.Series:
        return pd.Series(None, index=results.index)

    def get_entity_col(self, measure: str, results: pd.DataFrame) -> pd.Series:
        return pd.Series(None, index=results.index)

    def get_sub_entity_col(self, measure: str, results: pd.DataFrame) -> pd.Series:
        return pd.Series(None, index=results.index)
