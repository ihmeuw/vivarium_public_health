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
    lookup_tables: Iterable[LookupTable], necessary_columns: Iterable = ()
) -> List[str]:
    necessary_columns = set(necessary_columns)
    for lookup_table in lookup_tables:
        necessary_columns.update(set(lookup_table.key_columns))
        necessary_columns.update(set(lookup_table.parameter_columns))
    if "year" in necessary_columns:
        necessary_columns.remove("year")

    return list(necessary_columns)
