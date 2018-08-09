from datetime import datetime
import io
import os.path
import json
from typing import Sequence

import pandas as pd
import tables
from tables.nodes import filenode

import logging
_log = logging.getLogger(__name__)


class ArtifactException(Exception):
    pass


def parse_artifact_path_config(config):
    # NOTE: The artifact_path may be an absolute path or it may be relative to the location of the
    # config file.
    path_config = config.artifact.metadata('path')[-1]
    if path_config['source'] is not None:
        artifact_path = os.path.join(os.path.dirname(path_config['source']), path_config['value'])
    else:
        artifact_path = path_config['value']

    return artifact_path


class EntityKey(str):
    @property
    def type(self):
        return self.split('.')[0]

    @property
    def name(self):
        return self.split('.')[1] if len(self.split('.')) == 3 else ''

    @property
    def measure(self):
        return self.split('.')[-1]

    @property
    def group_prefix(self):
        return '/'+self.type if self.name else '/'

    @property
    def group_name(self):
        return self.name if self.name else self.type

    @property
    def group(self):
        return self.group_prefix + '/' + self.group_name if self.name else self.group_prefix + self.group_name

    def to_path(self, measure=None):
        measure = self.measure if not measure else measure
        return self.group + '/' + measure


class ArtifactManager:
    configuration_defaults = {
            'artifact': {
                'path': None,
            }
    }

    def setup(self, builder):
        artifact_path = parse_artifact_path_config(builder.configuration)
        column_filter, row_filter = get_default_filter(builder.configuration)
        self.artifact = Artifact(artifact_path, column_filter, row_filter)
        builder.event.register_listener('post_setup', lambda _: self.artifact.close())

    def load(self, entity_k, keep_age_group_edges=False, **column_filters):
        entity_key = EntityKey(entity_k)
        return self.artifact.load(entity_key, keep_age_group_edges, **column_filters)


class Artifact(pd.HDFStore):

    def __init__(self, path, default_column_filter, default_row_filter, **kwargs):
        super().__init__(path, **kwargs)
        self.draw = draw
        self.location = location
        self._cache = {}
        self._loading_start_time = None

    def load(self, entity_key, keep_age_group_edges=False, **column_filters):
        _log.debug(f"loading {entity_key}")
        cache_key = (entity_key, tuple(sorted(column_filters.items())))
        if cache_key in self._cache:
            _log.debug("    from cache")
        else:
            self._cache[cache_key] = self._uncached_load(entity_key, keep_age_group_edges, column_filters)
            if self._cache[cache_key] is None:
                raise ArtifactException(f"data for {entity_key} is not available. Check your model specification")

        return self._cache[cache_key]

    def _uncached_load(self, entity_key, keep_age_group_edges, column_filters):

        if entity_key.to_path() not in self._hdf:
            raise ArtifactException(f"{entity_key.to_path()} should be in {self.artifact_path}")

        node = self._hdf.get_node(entity_key.to_path())

        if isinstance(node, tables.earray.EArray):
            # This should be a json encoded document rather than a pandas dataframe
            fnode = filenode.open_node(node, 'r')
            document = json.load(fnode)
            fnode.close()
            return document
        else:
            columns = list(node.table.colindexes.keys())

        filter_terms, columns_to_remove = _setup_filter(columns, column_filters, self.location, self.draw)

        data = pd.read_hdf(self._hdf, entity_key.to_path(), where=filter_terms if filter_terms else None)

        if not keep_age_group_edges:
            # TODO: Should probably be using these age group bins rather than the midpoints but for now we use mids
            columns_to_remove |= {"age_group_start", "age_group_end"}
        columns_to_remove = columns_to_remove.intersection(columns)

        data = data.drop(columns=columns_to_remove)

        return data

    def write(self, key_components: Sequence[str], data):
        if data is None:
            pass
        inner_path = os.path.join(*key_components)

        if isinstance(data, (pd.DataFrame, pd.Series)):
            if data.empty:
                raise ValueError("Cannot persist empty dataset")
            data_columns = {"year", "location", "draw", "cause", "risk"}.intersection(data.columns)
            with pd.HDFStore(self.artifact_path, complevel=9, format="table") as store:
                store.put(inner_path, data, format="table", data_columns=data_columns)
        else:
            prefix = os.path.join(*key_components[:-2])
            store = tables.open_file(self.artifact_path, "a")
            if inner_path in store:
                store.remove_node(inner_path)
            try:
                store.create_group(prefix, key_components[-2], createparents=True)
            except tables.exceptions.NodeError as e:
                if "already has a child node" in str(e):
                    # The parent group already exists, which is fine
                    pass
                else:
                    raise

            fnode = filenode.new_node(store, where=os.path.join(*key_components[:-1]), name=key_components[-1])
            fnode.write(bytes(json.dumps(data), "utf-8"))
            fnode.close()
            store.close()

    def open(self):
        if self._hdf is None:
            self._loading_start_time = datetime.now()
            self._hdf = pd.HDFStore(self.artifact_path, mode='r')

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


def get_default_filter(configuration):
    draw = configuration.input_data.input_draw_number
    location = configuration.input_data.location


def _setup_filter(columns, column_filters, location, draw):
    terms = []
    default_filters = {
        'draw': draw,
    }
    column_filters = {**default_filters, **column_filters}
    for column, condition in column_filters.items():
        if column in columns:
            if not isinstance(condition, (list, tuple)):
                condition = [condition]
            for c in condition:
                terms.append(f"{column} = {c}")
        elif column not in default_filters:
            raise ValueError(f"Filtering by non-existant column '{column}'. Avaliable columns {columns}")
    columns_to_remove = set(column_filters.keys())
    if "location" not in column_filters and "location" in columns:
        terms.append(get_location_term(location))
        columns_to_remove.add("location")
    return terms, columns_to_remove


def get_location_term(location: str):
    template = "location == {quote_mark}{loc}{quote_mark} | location == {quote_mark}Global{quote_mark}"
    if "'" in location and '"' in location:  # Because who knows
        raise NotImplementedError(f"Unhandled location string {location}")
    elif "'" in location:  # Only because location names are weird
        quote_mark = '"'
    else:
        quote_mark = "'"

    return template.format(quote_mark=quote_mark, loc=location)


class ArtifactManagerInterface():
    def __init__(self, controller):
        self._controller = controller

    def load(self, entity_key, keep_age_group_edges=False, **column_filters):
        return self._controller.load(entity_key, keep_age_group_edges, **column_filters)
