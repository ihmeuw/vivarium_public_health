from datetime import datetime
import os.path

import pandas as pd
from tables.nodes import filenode

from vivarium.framework.time import _get_time_stamp

import logging
_log = logging.getLogger(__name__)

class Artifact:
    def setup(self, builder):
        self._loading_start_time = datetime.now()
        self.end_time = _get_time_stamp(builder.configuration.time.end)
        self.start_time = _get_time_stamp(builder.configuration.time.start)
        self.draw = builder.configuration.run_configuration.input_draw_number
        self.location = builder.configuration.input_data.location_id

        #NOTE: The artifact_path may be an absolute path or it may be relative to the location of the
        # config file.
        path_config = builder.configuration.input_data.metadata('artifact_path')[0]
        self.artifact_path = os.path.normpath(os.path.join(os.path.dirname(path_config['source']), path_config['value']))


        self.open()
        builder.event.register_listener('post_setup', lambda _: self.close())
        self._cache = {}

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
                    fnode = filenode.open_node(s._handle.get_node(group), 'r')
                    document = json.load(fnode)
                    fnode.close()
                    self._cache[cache_key] = document
                    return document
            except AttributeError:
                # This isn't a json node so move on
                pass
            #TODO: Is there a better way to get the columns without loading  much data?
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
                        if isinstance(c, str):
                            terms.append(f"{column} {c}")
                        else:
                            terms.append(f"{column} = {c}")
            columns_to_remove = set(column_filters.keys())
            if "location_id" not in column_filters and "location_id" in columns:
                #TODO I think this is a sign I should be handling queries differently
                terms.append(f"location_id == {self.location} | location_id == 1")
                columns_to_remove.add("location_id")
            data = pd.read_hdf(self._hdf, group, where=terms if terms else None)
            if not keep_age_group_edges:
                # TODO: Should probably be using these age group bins rather than the midpoints but for now we use mids
                columns_to_remove = columns_to_remove | {"age_group_start", "age_group_end"}
            columns_to_remove = columns_to_remove.intersection(columns)

            data = data.drop(columns=columns_to_remove)
            self._cache[(entity_path, None)] = data
        self._cache[cache_key] = data
        return data

    def open(self):
        self._hdf = pd.HDFStore(self.artifact_path, mode='r')

    def close(self):
        self._hdf.close()
        self._cache = {}
        _log.debug(f"Data loading took at most {datetime.now() - self._loading_start_time} seconds")
