"""A convenience wrapper around tables and pd.HDFStore."""
import json
from pathlib import Path
import typing
from typing import Any, List, Optional
import re

import pandas as pd
import tables
from tables.nodes import filenode

if typing.TYPE_CHECKING:
    from vivarium_public_health.dataset_manager import EntityKey

DEFAULT_COLUMNS = {"location", "draw"}


def touch(path: str, append: bool):
    """Creates an hdf file or wipes an existing file if necessary.

    If the file exists and append is not true, the existing file
    will be destroyed. If the file exists and append is true, touch
    will do nothing.

    Parameters
    ----------
    path :
        The string path to the hdf file.
    append :
        Whether an existing artifact should be preserved and appended to

    Raises
    ------
    FileNotFoundError
        If attempting to append to a non-existent file."""

    path = Path(path)

    if not path.is_file() and append:
        raise FileNotFoundError("You indicated you want to append to "
                                f"an existing artifact at {path} but no "
                                "such artifact exists. remove --append if "
                                "you wish to create a new artifact.")
    elif not append:
        with tables.open_file(str(path), mode='w'):
            pass


def write(path: str, entity_key: 'EntityKey', data: Any):
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
    if isinstance(data, pd.DataFrame):
        _write_data_frame(path, entity_key, data)
    else:
        _write_json_blob(path, entity_key, data)


def load(path: str, entity_key: 'EntityKey', filter_terms: Optional[List[str]]) -> Any:
    """Loads data from an hdf file.

    Parameters
    ----------
    path :
        The path to the hdf file to load the data from.
    entity_key :
        A representation of the internal hdf path where the data is located.
    filter_terms :
        A list of terms used to filter the data formatted in a way that is
        suitable for use with the `where` argument of `pd.read_hdf`. Only
        filters applying to existing columns in the data are used.

    Returns
    -------
        The data stored at the the given key in the hdf file.
    """
    with tables.open_file(path, mode='r') as file:
        node = file.get_node(entity_key.path)

        if isinstance(node, tables.earray.EArray):
            # This should be a json encoded document rather than a pandas dataframe
            with filenode.open_node(node, 'r') as file_node:
                data = json.load(file_node)
        else:
            filter_terms = _get_valid_filter_terms(filter_terms, node.table.colnames)
            data = pd.read_hdf(path, entity_key.path, where=filter_terms)

        return data


def remove(path: str, entity_key: 'EntityKey'):
    """Removes a piece of data from an hdf file.

    Parameters
    ----------
    path :
        The path to the hdf file to remove the data from.
    entity_key :
        A representation of the internal hdf path where the data is located.
    """
    with tables.open_file(path, mode='a') as file:
        file.remove_node(entity_key.path, recursive=True)


def get_keys(path: str) -> List[str]:
    """Gets key representation of all paths in an hdf file.

    Parameters
    ----------
    path :
        The path to the hdf file.

    Returns
    -------
        A list of key representations of the internal paths in the hdf.
    """
    with tables.open_file(path, mode='r') as file:
        keys = _get_keys(file.root)
    return keys


def _write_json_blob(path: str, entity_key: 'EntityKey', data: Any):
    """Writes a primitive python type or container as json to the hdf file at the given path."""
    with tables.open_file(path, "a") as store:

        if entity_key.group_prefix not in store:
            store.create_group('/', entity_key.type)

        if entity_key.group not in store:
            store.create_group(entity_key.group_prefix, entity_key.group_name)

        with filenode.new_node(store, where=entity_key.group, name=entity_key.measure) as fnode:
            fnode.write(bytes(json.dumps(data), "utf-8"))


def _write_data_frame(path: str, entity_key: 'EntityKey', data: pd.DataFrame):
    """Writes a pandas DataFrame or Series to the hdf file at the given path."""
    entity_path = entity_key.path
    if data.empty:
        raise ValueError("Cannot persist empty dataset")

    # Even though these get called data_columns, it's more correct to think of them
    # as the columns you can use to index into the raw data with. It's the subset of columns
    # that you can filter by without reading in a whole dataset.
    data_columns = DEFAULT_COLUMNS.intersection(data.columns)

    with pd.HDFStore(path, complevel=9, format="table") as store:
        store.put(entity_path, data, format="table", data_columns=data_columns)


def _get_keys(root: tables.node.Node, prefix: str='') -> List[str]:
    """Recursively formats the internal paths in an hdf file into a key format."""
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


def _get_node_name(node: tables.node.Node) -> str:
    """Gets the name of a node from its string representation."""
    node_string = str(node)
    node_path = node_string.split()[0]
    node_name = node_path.split('/')[-1]
    return node_name


def _get_valid_filter_terms(filter_terms, colnames):
    """Removes any filter terms referencing non-existent columns

    Parameters
    ----------
    filter_terms :
        A list of terms formatted so as to be used in the `where` argument of
        `pd.read_hdf`
    colnames :
        A list of column names present in the data that will be filtered

    Returns
    -------
        The list of valid filter terms (terms that do not reference any column
        not existing in the data). Returns none if the list is empty because
        the `where` argument doesn't like empty lists.
    """
    if not filter_terms:
        return None
    valid_terms = filter_terms.copy()
    for term in filter_terms:
        # first strip out all the parentheses - the where in read_hdf requires all references to be valid
        t = re.sub('[()]', '', term)
        # then split each condition out
        t = re.split('[&|]', t)
        # get the unique columns referenced by this term
        term_columns = set([re.split('[<=>\s]', i.strip())[0] for i in t])
        if not term_columns.issubset(colnames):
            valid_terms.remove(term)
    return valid_terms if valid_terms else None
