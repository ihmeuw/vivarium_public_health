"""
=========
Utilities
=========

This module contains utility classes and functions for use across
vivarium_public_health components.

"""
from typing import Union

import pandas as pd


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
        split = self.split('.')
        if len(split) != 2:
            raise ValueError(f'You must specify the entity as "entity_type.entity". You specified {self}.')
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
        split = self.split('.')
        if len(split) != 3:
            raise ValueError(
                f'You must specify the target as "affected_entity_type.affected_entity_name.affected_measure".'
                f'You specified {self}.')
        return split[0], split[1], split[2]


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
