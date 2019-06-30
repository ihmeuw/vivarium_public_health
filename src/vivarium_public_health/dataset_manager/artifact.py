"""
=================
The Data Artifact
=================

This module provides tools for interacting with data artifacts.

A data artifact is a hdf archive on disk intended to package up all data
relevant to a particular simulation. This module provides a class to wrap that
hdf file for convenient access and inspection.

"""
from collections import defaultdict
import logging
from typing import List, Dict, Any
from pathlib import Path
import re

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

        self.create_hdf_with_keyspace(path)
        self.path = path
        self._filter_terms = filter_terms
        self._draw_column_filter = _parse_draw_filters(filter_terms)
        self._cache = {}
        self._keys = Keys(self.path)

    @staticmethod
    def create_hdf_with_keyspace(path):
        if not Path(path).is_file():
            hdf.touch(path)
            hdf.write(path, EntityKey('metadata.keyspace'), ['metadata.keyspace'])

    @property
    def keys(self) -> List['EntityKey']:
        """A list of all the keys contained within the artifact."""
        return self._keys.to_list()

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
            data = hdf.load(self.path, entity_key, self._filter_terms, self._draw_column_filter)

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

    def replace(self, entity_key: str, data: Any):
        """Replaces the data in the artifact at the provided key with the prov.

        Parameters
        ----------
        entity_key :
            The key for which the data should be overwritten.
        data :
            The data to write. Accepted formats are ``pandas`` Series or DataFrames
            or standard python types and containers.

        Raises
        ------
        ArtifactException :
            If the provided key does not already exist in the artifact.
        """
        e_key = EntityKey(entity_key)

        if e_key not in self.keys:
            raise ArtifactException(f'Trying to replace non-existent key {e_key} in artifact.')
        self.remove(entity_key)
        self.write(entity_key, data)

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


class Keys:
    """A convenient wrapper around the keyspace which makes easier for Artifact
     to maintain its keyspace when EntityKey is added or removed.
     With the artifact_path, Keys object is initialized when the Artifact is
     initialized """

    keyspace_node = EntityKey('metadata.keyspace')

    def __init__(self, artifact_path: str):
        self.artifact_path = artifact_path
        self._keys = [str(k) for k in hdf.load(self.artifact_path, EntityKey('metadata.keyspace'), None, None)]

    def append(self, new_key: EntityKey):
        """ Whenever the artifact gets a new key and new data, append is called to
        remove the old keyspace and to write the updated keyspace"""

        self._keys.append(str(new_key))
        hdf.remove(self.artifact_path, self.keyspace_node)
        hdf.write(self.artifact_path, self.keyspace_node, self._keys)

    def remove(self, removing_key: EntityKey):
        """ Whenever the artifact removes a key and data, remove is called to
        remove the key from keyspace and write the updated keyspace."""

        self._keys.remove(str(removing_key))
        hdf.remove(self.artifact_path, self.keyspace_node)
        hdf.write(self.artifact_path, self.keyspace_node, self._keys)

    def to_list(self) -> List[EntityKey]:
        """A list of all the EntityKeys in the associated artifact."""

        return [EntityKey(k) for k in self._keys]

    def __contains__(self, item):
        return item in self._keys


def _parse_draw_filters(filter_terms):
    """Given a list of filter terms, parse out any related to draws and convert
    to the list of column names. Also include 'value' column for compatibility
    with data that is long on draws."""
    columns = None

    if filter_terms:
        draw_terms = []
        for term in filter_terms:
            # first strip out all the parentheses
            t = re.sub('[()]', '', term)
            # then split each condition out
            t = re.split('[&|]', t)
            # then split condition to see if it relates to draws
            split_term = [re.split('([<=>in])', i) for i in t]
            draw_terms.extend([t for t in split_term if t[0].strip() == 'draw'])

        if len(draw_terms) > 1:
            raise ValueError(f'You can only supply one filter term related to draws. '
                             f'You supplied {filter_terms}, {len(draw_terms)} of which pertain to draws.')

        if draw_terms:
            # convert term to columns
            term = [s.strip() for s in draw_terms[0] if s.strip()]
            if len(term) == 4 and term[1].lower() == 'i' and term[2].lower() == 'n':
                draws = [int(d) for d in term[-1][1:-1].split(',')]
            elif (len(term) == 4 and term[1] == term[2] == '=') or (len(term) == 3 and term[1] == '='):
                draws = [int(term[-1])]
            else:
                raise NotImplementedError(f'The only supported draw filters are =, ==, or in. '
                                          f'You supplied {"".join(term)}.')

            columns = [f'draw_{n}' for n in draws] + ['value']

    return columns
