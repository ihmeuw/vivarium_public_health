"""
=========
Utilities
=========

This module contains utility classes and functions for use across
vivarium_public_health components.

"""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import pandas as pd
from vivarium.framework.lookup import LookupTable, ScalarValue
from vivarium.framework.results import METRICS_COLUMN, StratifiedObserver


class EntityString(str):
    """Convenience class for representing entities as strings."""

    def __init__(self, entity):
        super().__init__()
        self._type, self._name = self.split_entity()

    @property
    def type(self):
        return self._type

    @property
    def name(self):
        return self._name

    def split_entity(self):
        split = self.split(".")
        if len(split) != 2:
            raise ValueError(
                f'You must specify the entity as "entity_type.entity". You specified {self}.'
            )
        return split[0], split[1]


class TargetString(str):
    """Convenience class for representing risk targets as strings."""

    def __init__(self, target):
        super().__init__()
        self._type, self._name, self._measure = self.split_target()

    @property
    def type(self):
        return self._type

    @property
    def name(self):
        return self._name

    @property
    def measure(self):
        return self._measure

    def split_target(self):
        split = self.split(".")
        if len(split) != 3:
            raise ValueError(
                f"You must specify the target as "
                f'"affected_entity_type.affected_entity_name.affected_measure". '
                f"You specified {self}."
            )
        return split[0], split[1], split[2]


def to_snake_case(string: str) -> str:
    return string.lower().replace(" ", "_").replace("-", "_")


DAYS_PER_YEAR = 365.25
DAYS_PER_MONTH = DAYS_PER_YEAR / 12


def to_time_delta(span_in_days: Union[int, float, str]):
    span_in_days = float(span_in_days)
    days, remainder = span_in_days // 1, span_in_days % 1
    hours, remainder = (24 * remainder) // 24, (24 * remainder) % 24
    minutes = (60 * remainder) // 60
    return pd.Timedelta(days=days, hours=hours, minutes=minutes)


def to_years(time: pd.Timedelta) -> float:
    """Converts a time delta to a float for years."""
    return time / pd.Timedelta(days=DAYS_PER_YEAR)


def is_non_zero(data: Union[Iterable[ScalarValue], ScalarValue, pd.DataFrame]) -> bool:
    if isinstance(data, pd.DataFrame):
        attribute_sum = data.value.sum()
    elif isinstance(data, Iterable):
        attribute_sum = sum(data)
    else:
        attribute_sum = data

    return attribute_sum != 0.0


def get_lookup_columns(
    lookup_tables: List[LookupTable], necessary_columns: Iterable = ()
) -> List[str]:
    necessary_columns = set(necessary_columns)
    for lookup_table in lookup_tables:
        necessary_columns.update(lookup_table.key_columns)
        necessary_columns.update(lookup_table.parameter_columns)
    if "year" in necessary_columns:
        necessary_columns.remove("year")

    return list(necessary_columns)


def get_index_columns_from_lookup_configuration(
    lookup_configuration: Dict[str, List[str]]
) -> List[str]:
    index_columns = []
    for column in lookup_configuration["continuous_columns"]:
        start_column = f"{column}_start"
        end_column = f"{column}_end"
        index_columns.extend([start_column, end_column])
    for column in lookup_configuration["categorical_columns"]:
        index_columns.append(column)
    return index_columns


def write_dataframe_to_csv(
    observer: StratifiedObserver,
    measure: str,
    results: pd.DataFrame,
    extra_cols: Optional[Dict[str, Any]] = {},
) -> None:
    """Utility function for observation 'report' methods to write pd.DataFrames to csv"""
    results_dir = Path(observer.results_dir)
    # Add extra cols
    col_mapper = {"measure": measure}
    col_mapper.update(extra_cols)
    col_mapper.update(
        {"random_seed": observer.random_seed, "input_draw": observer.input_draw}
    )
    for col, val in col_mapper.items():
        if val is not None:
            results[col] = val
    # Sort the columns such that the stratifications (index) are first
    # and METRICS_COLUMN is last and sort the rows by the stratifications.
    other_cols = [c for c in results.columns if c != METRICS_COLUMN]
    results = results[other_cols + [METRICS_COLUMN]].sort_index().reset_index()

    # Concat and save
    results_file = results_dir / f"{measure}.csv"
    if not results_file.exists():
        results.to_csv(results_file, index=False)
    else:
        results.to_csv(results_dir / f"{measure}.csv", index=False, mode="a", header=False)
