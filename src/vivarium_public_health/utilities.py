"""
=========
Utilities
=========

This module contains utility classes and functions for use across
vivarium_public_health components.

"""


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
