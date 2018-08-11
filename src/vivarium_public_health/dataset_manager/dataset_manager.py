import logging
import os.path

from vivarium.config_tree import ConfigTree
from .artifact import Artifact, EntityKey

_log = logging.getLogger(__name__)


class ArtifactManagerInterface():
    def __init__(self, controller):
        self._controller = controller

    def load(self, entity_key, keep_age_group_edges=False, **column_filters):
        return self._controller.load(entity_key, keep_age_group_edges, **column_filters)


class ArtifactManager:
    configuration_defaults = {
            'artifact': {
                'path': None,
            }
    }

    def setup(self, builder):
        artifact_path = parse_artifact_path_config(builder.configuration)
        draw = builder.configuration.input_data.input_draw_number,
        location = builder.configuration.input_data.location
        base_filter_terms = [f'draw == {draw}', get_location_term(location)]
        self.artifact = self._load_artifact(artifact_path, base_filter_terms)
        builder.event.register_listener('post_setup', lambda _: self.artifact.close())

    def _load_artifact(self, artifact_path, base_filter_terms):
        return Artifact(artifact_path, base_filter_terms)

    def load(self, entity_k, keep_age_group_edges=False, **column_filters):
        entity_key = EntityKey(entity_k)
        data = self.artifact.load(entity_key)
        return filter_data(data, keep_age_group_edges, **column_filters)


def filter_data(data, keep_age_group_edges, **column_filters):
    for column, condition in column_filters.items():
        if column in data.columns:
            if not isinstance(condition, (list, tuple)):
                condition = [condition]
            for c in condition:
                data = data.loc[f"{column} = {c}", :]
        else:
            raise ValueError(f"Filtering by non-existent column '{column}'. Available columns {columns}")

    columns_to_remove = set(list(column_filters.keys()) + ['draw', 'location'])
    columns_to_remove.add("location")
    if not keep_age_group_edges:
        # TODO: Should probably be using these age group bins rather than the midpoints but for now we use mids
        columns_to_remove |= {"age_group_start", "age_group_end"}

    columns_to_remove = columns_to_remove.intersection(data.columns)
    return data.drop(columns=columns_to_remove)


def get_location_term(location: str):
    template = "location == {quote_mark}{loc}{quote_mark} | location == {quote_mark}Global{quote_mark}"
    if "'" in location and '"' in location:  # Because who knows
        raise NotImplementedError(f"Unhandled location string {location}")
    elif "'" in location:
        quote_mark = '"'
    else:
        quote_mark = "'"

    return template.format(quote_mark=quote_mark, loc=location)


def parse_artifact_path_config(config: ConfigTree) -> str:
    """Gets the path to the data artifact from the simulation configuration.

    Parameters
    ----------
    config :
        The configuration block of the simulation model specification containing the artifact path.

    Returns
    -------
        The path to the data artifact.
    """
    # NOTE: The artifact_path may be an absolute path or it may be relative to the location of the config file.
    path_config = config.artifact.metadata('path')[-1]
    if path_config['source'] is not None:
        artifact_path = os.path.join(os.path.dirname(path_config['source']), path_config['value'])
    else:
        artifact_path = path_config['value']

    return artifact_path

