"""This module provides tools for interacting with data artifacts.

A data artifact is a hdf archive on disk intended to package up all data relevant
to a particular simulation. This module provides a class to wrap that hdf file for
convenient access and inspection.
"""
from collections import defaultdict
import logging
from typing import List, Dict, Any

from vivarium_public_health.dataset_manager import hdf

_log = logging.getLogger(__name__)


class ArtifactException(Exception):
    """Exception raise for inconsistent use of the data artifact."""
    pass


class Artifact:
    """An interface for interacting with ``vivarium`` hdf artifacts."""

    def __init__(self, path: str, filter_terms: List[str]=None):
        """
        Parameters
        ----------
        path :
            The path to the hdf artifact.
        filter_terms :
            A set of terms suitable for usage with the ``where`` kwarg for ``pd.read_hdf``
        """

        self.path = path
        self._filter_terms = filter_terms

        self._cache = {}
        self._keys = [EntityKey(k) for k in hdf.get_keys(self.path)]

    @property
    def keys(self) -> List['EntityKey']:
        """A list of all the keys contained within the artifact."""
        return self._keys

    @property
    def filter_terms(self) -> List[str]:
        return self._filter_terms

    def load(self, entity_key: str) -> Any:
        """Loads the data associated with provided EntityKey.

        Parameters
        ----------
        entity_key :
            The key associated with the expected data.

        Returns
        -------
            The expected data. Will either be a standard Python object or a
            ``pandas`` series or dataframe.

        Raises
        ------
        ArtifactException :
            If the provided key is not in the artifact.
        """
        entity_key = EntityKey(entity_key)
        if entity_key not in self.keys:
            raise ArtifactException(f"{entity_key} should be in {self.path}.")

        if entity_key not in self._cache:
            data = hdf.load(self.path, entity_key, self._filter_terms)

            assert data is not None, f"Data for {entity_key} is not available. Check your model specification."

            self._cache[entity_key] = data

        return self._cache[entity_key]

    def write(self, entity_key: str, data: Any):
        """Writes the provided data into the artifact and binds it to the provided key.

        Parameters
        ----------
        entity_key :
            The key associated with the provided data.
        data :
            The data to write. Accepted formats are ``pandas`` Series or DataFrames
            or standard python types and containers.

        Raises
        ------
        ArtifactException :
            If the provided key already exists in the artifact.
        """
        entity_key = EntityKey(entity_key)
        if entity_key in self.keys:
            raise ArtifactException(f'{entity_key} already in artifact.')
        elif data is None:
            pass
        else:
            self._keys.append(entity_key)
            hdf.write(self.path, entity_key, data)

    def remove(self, entity_key: str):
        """Removes data associated with the provided key from the artifact.

        Parameters
        ----------
        entity_key :
            The key associated with the data to remove.

        Raises
        ------
        ArtifactException :
            If the key is not present in the artifact."""
        entity_key = EntityKey(entity_key)
        if entity_key not in self.keys:
            raise ArtifactException(f'Trying to remove non-existent key {entity_key} from artifact.')

        self._keys.remove(entity_key)
        if entity_key in self._cache:
            self._cache.pop(entity_key)
        hdf.remove(self.path, entity_key)

    def clear_cache(self):
        """Clears the artifact's cache.

        The artifact will cache data in memory to improve performance for repeat access.
        """
        self._cache = {}

    def __iter__(self):
        return iter(self.keys)

    def __contains__(self, item: str):
        return EntityKey(item) in self.keys

    def __repr__(self):
        return f"Artifact(keys={self.keys})"

    def __str__(self):
        key_tree = _to_tree(self.keys)
        out = "Artifact containing the following keys:\n"
        for root, children in key_tree.items():
            out += f'{root}\n'
            for child, grandchildren in children.items():
                out += f'\t{child}\n'
                for grandchild in grandchildren:
                    out += f'\t\t{grandchild}\n'
        return out


class EntityKey(str):
    """A convenience wrapper around the keys used by the simulation to look up entity data in the artifact."""

    def __init__(self, key):
        elements = [e for e in key.split('.') if e]
        if len(elements) not in [2, 3] or len(key.split('.')) != len(elements):
            raise ValueError(f'Invalid format for EntityKey: {key}. '
                             'Acceptable formats are "type.name.measure" and "type.measure"')
        super().__init__()

    @property
    def type(self) -> str:
        """The type of the entity represented by the key."""
        return self.split('.')[0]

    @property
    def name(self) -> str:
        """The name of the entity represented by the key"""
        return self.split('.')[1] if len(self.split('.')) == 3 else ''

    @property
    def measure(self) -> str:
        """The measure associated with the data represented by the key."""
        return self.split('.')[-1]

    @property
    def group_prefix(self) -> str:
        """The hdf group prefix for the key."""
        return '/'+self.type if self.name else '/'

    @property
    def group_name(self) -> str:
        """The hdf group name for the key."""
        return self.name if self.name else self.type

    @property
    def group(self) -> str:
        """The full path to the group for this key."""
        return self.group_prefix + '/' + self.group_name if self.name else self.group_prefix + self.group_name

    @property
    def path(self) -> str:
        """Converts this entity key to its hdfstore path.

        Returns
        -------
            The full hdfstore path for this key.
        """
        return self.group + '/' + self.measure

    def with_measure(self, measure: str) -> 'EntityKey':
        """Gets another EntityKey with the same type and name but a different measure.

        Parameters
        ----------
        measure :
            The measure to replace this key's measure with.

        Returns
        -------
            A new EntityKey with the updated measure.
        """
        if self.name:
            return EntityKey(f'{self.type}.{self.name}.{measure}')
        else:
            return EntityKey(f'{self.type}.{measure}')

    def __eq__(self, other: 'EntityKey') -> bool:
        return isinstance(other, EntityKey) and str(self) == str(other)

    def __hash__(self):
        return hash(str(self))

    def __repr__(self) -> str:
        return f'EntityKey({str(self)})'


def _to_tree(keys: List[EntityKey]) -> Dict[str, Dict[str, List[str]]]:
    out = defaultdict(lambda: defaultdict(list))
    for k in keys:
        if k.name:
            out[k.type][k.name].append(k.measure)
        else:
            out[k.type][k.measure] = []
    return out
