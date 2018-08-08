from datetime import datetime
import io
import os.path
import json

import pandas as pd
from tables.nodes import filenode

from vivarium.framework.time import get_time_stamp

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


class ArtifactManager:
    configuration_defaults = {
            'artifact': {
                'path': None,
            }
    }

    def setup(self, builder):
        end_time = get_time_stamp(builder.configuration.time.end)
        start_time = get_time_stamp(builder.configuration.time.start)
        draw = builder.configuration.input_data.input_draw_number
        location = builder.configuration.input_data.location

        artifact_path = parse_artifact_path_config(builder.configuration)
        self.artifact = Artifact(artifact_path, start_time, end_time, draw, location)

        self.artifact.open()
        builder.event.register_listener('post_setup', lambda _: self.artifact.close())

    def load(self, entity_path, keep_age_group_edges=False, **column_filters):
        return self.artifact.load(entity_path, keep_age_group_edges, **column_filters)


class Artifact:
    def __init__(self, path, start_time, end_time, draw, location):
        self.artifact_path = path
        self.start_time = start_time
        self.end_time = end_time
        self.draw = draw
        self.location = location

        self._cache = {}
        self._hdf = None

        self._loading_start_time = None

    def load(self, entity_path, keep_age_group_edges=False, **column_filters):
        _log.debug(f"loading {entity_path}")
        cache_key = (entity_path, tuple(sorted(column_filters.items())))
        if cache_key in self._cache:
            _log.debug("    from cache")
        else:
            self._cache[cache_key] = self._uncached_load(entity_path, keep_age_group_edges, column_filters)
            if self._cache[cache_key] is None:
                raise ArtifactException(f"data for {entity_path} is not available. Check your model specification")

        return self._cache[cache_key]

    def _uncached_load(self, entity_path, keep_age_group_edges, column_filters):
        group = '/'+entity_path.replace('.', '/')

        if group not in self._hdf:
            raise ArtifactException(f"{group} should be in {self.artifact_path}")

        node = self._hdf.get_node(group)
        if "NODE_TYPE" in dir(node._v_attrs) and node.get_attr("NODE_TYPE") == "file":
            # This should be a json encoded document rather than a pandas dataframe
            fnode = filenode.open_node(self._hdf.get_node(group), 'r')
            document = json.load(fnode)
            fnode.close()
            return document
        else:
            columns = list(self._hdf.get_node(group).table.colindexes.keys())

        filter_terms, columns_to_remove = _setup_filter(columns, column_filters, self.location, self.draw)

        data = pd.read_hdf(self._hdf, group, where=filter_terms if filter_terms else None)
        if not keep_age_group_edges:
            # TODO: Should probably be using these age group bins rather than the midpoints but for now we use mids
            columns_to_remove |= {"age_group_start", "age_group_end"}
        columns_to_remove = columns_to_remove.intersection(columns)

        data = data.drop(columns=columns_to_remove)

        return data

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
        for child in self._hdf.root._v_children:
            result.write(f"{child}\n")
            for sub_child in self._hdf.get_node(child)._v_children:
                result.write(f"\t{sub_child}\n")
        return result.getvalue()


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

    def load(self, entity_path, keep_age_group_edges=False, **column_filters):
        return self._controller.load(entity_path, keep_age_group_edges, **column_filters)
