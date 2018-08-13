"""This module contains the data artifact"""
import io
import logging
import json

import pandas as pd
import tables
from tables.nodes import filenode


_log = logging.getLogger(__name__)


class ArtifactException(Exception):
    """Exception raise for inconsistent use of the data artifact."""
    pass


class Artifact:
    default_columns = {"year", "location", "draw", "cause", "risk"}

    def __init__(self, path, filter_terms=None):
        self.path = path
        self.filter_terms = filter_terms

        self._hdf = None
        self._cache = {}
        self._keys = self._build_keys()

    def load(self, entity_key):
        _log.debug(f"loading {entity_key}")

        if entity_key in self._cache:
            _log.debug("    from cache")
        else:
            self._cache[entity_key] = self._uncached_load(entity_key)
            if self._cache[entity_key] is None:
                raise ArtifactException(f"data for {entity_key} is not available. Check your model specification")

        return self._cache[entity_key]

    def _uncached_load(self, entity_key):
        if entity_key.to_path() not in self._hdf:
            raise ArtifactException(f"{entity_key.to_path()} should be in {self.path}")

        node = self._hdf.get_node(entity_key.to_path())

        if isinstance(node, tables.earray.EArray):
            # This should be a json encoded document rather than a pandas dataframe
            fnode = filenode.open_node(node, 'r')
            data = json.load(fnode)
            fnode.close()
        else:
            data = pd.read_hdf(self._hdf, entity_key.to_path(), where=self.filter_terms)
        return data

    def write(self, entity_key, data):
        if data is None:
            pass
        elif isinstance(data, (pd.DataFrame, pd.Series)):
            self._write_data_frame(entity_key, data)
        else:
            self._write_json_blob(entity_key, data)

    def _write_data_frame(self, entity_key, data):
        entity_path = entity_key.to_path()
        if data.empty:
            raise ValueError("Cannot persist empty dataset")
        data_columns = Artifact.default_columns.intersection(data.columns)
        with pd.HDFStore(self.path, complevel=9, format="table") as store:
            store.put(entity_path, data, format="table", data_columns=data_columns)

    def _write_json_blob(self, entity_key, data):
        entity_path = entity_key.to_path()
        store = tables.open_file(self.path, "a")
        if entity_path in store:
            store.remove_node(entity_path)
        try:
            store.create_group(entity_key.group_prefix, entity_key.group_name, createparents=True)
        except tables.exceptions.NodeError as e:
            if "already has a child node" in str(e):
                # The parent group already exists, which is fine
                pass
            else:
                raise

        fnode = filenode.new_node(store, where=entity_key.group, name=entity_key.measure)
        fnode.write(bytes(json.dumps(data), "utf-8"))
        fnode.close()
        store.close()

    def clear_cache(self):
        self._cache = {}

    def _open(self, mode):
        if self._hdf is None:
            self._hdf = tables.open_file(self.path, mode=mode)
        else:
            raise ArtifactException("Opening already open artifact")

    def _close(self):
        if self._hdf is not None:
            self._hdf.close()
            self._hdf = None
        else:
            raise ArtifactException("Closing already closed artifact")

    def _build_keys(self):
        self._open('r')
        root = self._hdf.root



    def summary(self):
        result = io.StringIO()
        for child in self._hdf.root:
            result.write(f"{child}\n")
            for sub_child in child:
                result.write(f"\t{sub_child}\n")
        return result.getvalue()


def get_keys(root: tables.node.Node, prefix):
    keys = []
    for child in root:
        child_name = get_node_name(child)
        if isinstance(child, tables.earray.EArray):  # This is the last node
            keys.append(f'{prefix}.{child_name}')
        elif isinstance(child, tables.table.Table):  # Parent was the last node
            keys.append(prefix)
        else:
            keys.extend(get_keys(child, f'{prefix}.{child_name}'))
    return keys






def get_node_name(node: tables.node.Node):
    node_string = str(node)
    node_path = node_string.split()[0]
    node_name = node_path.split('/')[-1]
    return node_name


class EntityKey(str):
    """A convenience wrapper around the keys used by the simulation to look up entity data in the artifact."""

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

    def with_measure(self, measure: str) -> 'EntityKey':
        if self.name:
            return EntityKey(f'{self.type}.{self.name}.{measure}')
        else:
            return EntityKey(f'{self.type}.{measure}')

    def to_path(self, measure: str=None) -> str:
        """Converts this entity key to its hdfstore path.

        Parameters
        ----------
        measure :
            An override for this key's measure, the leaf of the hdfstore path.

        Returns
        -------
            The full hdfstore path for this key.
        """
        measure = self.measure if not measure else measure
        return self.group + '/' + measure
