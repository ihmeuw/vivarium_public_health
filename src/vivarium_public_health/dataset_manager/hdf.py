"""A convenience wrapper around tables and pd.HDFStore."""
import json
import typing
from typing import Any

import pandas as pd
import tables
from tables.nodes import filenode

if typing.TYPE_CHECKING:
    from vivarium_public_health.dataset_manager import EntityKey

DEFAULT_COLUMNS = {"year", "location", "draw", "cause", "risk"}


def write(path: str, entity_key: EntityKey, data: Any):
    """Writes data to the hdf file at the given path to the given key.

    Parameters
    ----------
    path :
        The path to the hdf file to write to.
    entity_key :
        A representation of the internal hdf path where we want to write the data.
    data :
        The data to write
    """
    if isinstance(data, (pd.DataFrame, pd.Series)):
        _write_data_frame(path, entity_key, data)
    else:
        _write_json_blob(path, entity_key, data)


def load(path, entity_key, filter_terms):
    """Loads data from an hdf file.

    Parameters
    ----------
    path : str
        The path to the hdf file to load the data from.
    entity_key : EntityKey
        A representation of the internal hdf path where the data is located.
    filter_terms : List[str]
        A list of terms used to filter the data formatted in a way that is
        suitable for use with the `where` argument of `pd.read_hdf`.

    Returns
    -------
    Any :
        The data stored at the the given key in the hdf file.
    """
    file = tables.open_file(path, mode='r')
    node = file.get_node(entity_key.path)

    if isinstance(node, tables.earray.EArray):
        # This should be a json encoded document rather than a pandas dataframe
        fnode = filenode.open_node(node, 'r')
        data = json.load(fnode)
        fnode.close()
        file.close()
    else:
        data = pd.read_hdf(path, entity_key.path, where=filter_terms)

    return data


def remove(path, entity_key):
    """Removes a piece of data from an hdf file.

    Parameters
    ----------
    path : str
        The path to the hdf file to remove the data from.
    entity_key : EntityKey
        A representation of the internal hdf path where the data is located.
    """
    with tables.open_file(path, mode='a') as file:
        file.remove_node(entity_key.path, recursive=True)


def get_keys(path):
    """Gets key representation of all paths in an hdf file.

    Parameters
    ----------
    path : str
        The path to the hdf file.

    Returns
    -------
    List[str] :
        A list of key representations of the internal paths in the hdf.
    """
    with tables.open_file(path, mode='r') as file:
        keys = _get_keys(file.root)
    return keys


def _write_json_blob(path, entity_key, data):
        entity_path = entity_key.path

        with tables.open_file(path, "a") as store:
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

            with filenode.new_node(store, where=entity_key.group, name=entity_key.measure) as fnode:
                fnode.write(bytes(json.dumps(data), "utf-8"))

        store.close()


def _write_data_frame(path, entity_key, data):
    entity_path = entity_key.path
    if data.empty:
        raise ValueError("Cannot persist empty dataset")

    data_columns = DEFAULT_COLUMNS.intersection(data.columns)

    with pd.HDFStore(path, complevel=9, format="table") as store:
        store.put(entity_path, data, format="table", data_columns=data_columns)


def _get_keys(root, prefix=''):
    keys = []
    for child in root:
        child_name = _get_node_name(child)
        if isinstance(child, tables.earray.EArray):  # This is the last node
            keys.append(f'{prefix}.{child_name}')
        elif isinstance(child, tables.table.Table):  # Parent was the last node
            keys.append(prefix)
        else:
            new_prefix = f'{prefix}.{child_name}' if prefix else child_name
            keys.extend(_get_keys(child, new_prefix))

    # Clean up some weird meta groups that get written with dataframes.
    keys = [k for k in keys if '.meta.' not in k]
    return keys


def _get_node_name(node: tables.node.Node):
    node_string = str(node)
    node_path = node_string.split()[0]
    node_name = node_path.split('/')[-1]
    return node_name
