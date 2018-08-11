from datetime import datetime
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
        self._cache = {}

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

    def write(self, entity_key, data, measure=None):
        entity_path = entity_key.to_path(measure)
        if data is None:
            pass
        elif isinstance(data, (pd.DataFrame, pd.Series)):
            self._write_data_frame(entity_key, measure, data)
        else:
            self._write_json_blob(entity_key, measure, data)

    def _write_data_frame(self, entity_key, measure, data):
        entity_path = entity_key.to_path(measure)
        if data.empty:
            raise ValueError("Cannot persist empty dataset")
        data_columns = Artifact.default_columns.intersection(data.columns)
        with pd.HDFStore(self.path, complevel=9, format="table") as store:
            store.put(entity_path, data, format="table", data_columns=data_columns)

    def _write_json_blob(self, entity_key, measure, data):
        entity_path = entity_key.to_path(measure)
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

        fnode = filenode.new_node(store, where=entity_key.group, name=measure)
        fnode.write(bytes(json.dumps(data), "utf-8"))
        fnode.close()
        store.close()

    def open(self):
        if self._hdf is None:
            self._loading_start_time = datetime.now()
            self._hdf = pd.HDFStore(self.path, mode='r')

        else:
            raise ArtifactException("Opening already open artifact")

    def close(self):
        if self._hdf is not None:
            self._hdf.close()
            self._hdf = None
            self._cache = {}
            _log.debug(f"Data loading took at most {datetime.now() - self._loading_start_time} seconds")
        else:
            raise ArtifactException("Closing already closed artifact")

    def summary(self):
        result = io.StringIO()
        for child in self._hdf.root:
            result.write(f"{child}\n")
            for sub_child in child:
                result.write(f"\t{sub_child}\n")
        return result.getvalue()


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

    def to_path(self, measure: str=None) -> str:
        measure = self.measure if not measure else measure
        return self.group + '/' + measure
