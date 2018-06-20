from datetime import datetime
import io
import os.path
import json

import pandas as pd
from tables.nodes import filenode

from vivarium.framework.time import _get_time_stamp

import logging
_log = logging.getLogger(__name__)


class ArtifactException(Exception):
    pass


class ArtifactManager:
    configuration_defaults = {
            'artifact': {
                'path': None,
            }
    }

    def setup(self, builder):
        end_time = _get_time_stamp(builder.configuration.time.end)
        start_time = _get_time_stamp(builder.configuration.time.start)
        draw = builder.configuration.input_data.input_draw_number
        location = builder.configuration.input_data.location

        # NOTE: The artifact_path may be an absolute path or it may be relative to the location of the
        # config file.
        path_config = builder.configuration.artifact.metadata('path')[-1]
        if path_config['source'] is not None:
            artifact_path = os.path.join(os.path.dirname(path_config['source']), path_config['value'])
        else:
            artifact_path = path_config['value']
        self.artifact = Artifact()

        self.artifact.open(artifact_path, start_time, end_time, draw, location)
        builder.event.register_listener('post_setup', lambda _: self.artifact.close())

    def load(self, entity_path, keep_age_group_edges=False, **column_filters):
        return self.artifact.load(entity_path, keep_age_group_edges, **column_filters)


class Artifact:
    def __init__(self):
        self.artifact_path = None
        self.start_time = None
        self.end_time = None
        self.draw = None
        self.location = None

        self._cache = {}
        self._hdf = None

        self._loading_start_time = None

    def load(self, entity_path, keep_age_group_edges=False, **column_filters):
        _log.debug(f"loading {entity_path}")
        cache_key = (entity_path, tuple(sorted(column_filters.items())))
        if cache_key in self._cache:
            _log.debug("    from cache")
            return self._cache[cache_key]
        else:
            group = '/'+entity_path.replace('.','/')
            try:
                node_type = self._hdf._handle.get_node_attr(group, "NODE_TYPE")
                if node_type == "file":
                    # This should be a json encoded document rather than a pandas dataframe
                    fnode = filenode.open_node(self._hdf._handle.get_node(group), 'r')
                    document = json.load(fnode)
                    fnode.close()
                    self._cache[cache_key] = document
                    return document
            except AttributeError:
                # This isn't a json node so move on
                pass
            # TODO: Is there a better way to get the columns without loading  much data?
            try:
                columns = self._hdf.select(group, stop=1).columns
            except KeyError:
                return None
            terms = []
            default_filters = {
                'draw': self.draw,
            }
            default_filters.update(column_filters)
            column_filters = default_filters
            for column, condition in column_filters.items():
                if column in columns:
                    if not isinstance(condition, (list, tuple)):
                        condition = [condition]
                    for c in condition:
                        terms.append(f"{column} = {c}")
            columns_to_remove = set(column_filters.keys())
            if "location" not in column_filters and "location" in columns:
                # TODO I think this is a sign I should be handling queries differently
                terms.append(f"location == {self.location} | location == 'Global'")
                columns_to_remove.add("location")
            data = pd.read_hdf(self._hdf, group, where=terms if terms else None)
            if not keep_age_group_edges:
                # TODO: Should probably be using these age group bins rather than the midpoints but for now we use mids
                columns_to_remove |= {"age_group_start", "age_group_end"}
            columns_to_remove = columns_to_remove.intersection(columns)

            data = data.drop(columns=columns_to_remove)
            self._cache[(entity_path, None)] = data
        self._cache[cache_key] = data
        return data

    def open(self, path, start_time, end_time, draw, location):
        if self._hdf is None:
            self.artifact_path = path
            self.start_time = start_time
            self.end_time = end_time
            self.draw = draw
            self.location = location
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
        for child in self._hdf._handle.root._v_children:
            result.write(f"{child}\n")
            for sub_child in getattr(self._hdf._handle.root, child)._v_children:
                result.write(f"\t{sub_child}\n")
        return result.getvalue()


class ArtifactManagerInterface():
    def __init__(self, controller):
        self._controller = controller

    def load(self, entity_path, keep_age_group_edges=False, **column_filters):
        return self._controller.load(entity_path, keep_age_group_edges, **column_filters)
