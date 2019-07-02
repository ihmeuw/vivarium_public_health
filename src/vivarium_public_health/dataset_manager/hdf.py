"""
=============
HDF Interface
=============

A convenience wrapper around tables and pd.HDFStore.

"""
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


def touch(path: str):
    """Creates an hdf file or wipes an existing file if necessary.
    If the given path is proper to create a hdf file, it creates a new hdf file.

    Parameters
    ----------
    path :
        The string path to the hdf file.

    Raises
    ------
    ValueError
        If the non-proper path is given to create a hdf file.
    """
    path = Path(path)
    if not path.suffix == '.hdf':
        raise ValueError(f'You provided path: {str(path)} not valid for creating hdf file.')

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
    # TODO: should we support writing index objects directly? If we do, .reset_index() in the builder etc. is
    # problematic. Currently not supporting.
    else:
        _write_json_blob(path, entity_key, data)


def load(path: str, entity_key: 'EntityKey', filter_terms: Optional[List[str]], column_filters: Optional[List[str]]) -> Any:
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
            with pd.HDFStore(path, complevel=9, mode='r') as store:
                metadata = store.get_storer(entity_key.path).attrs.metadata  # NOTE: must use attrs. write this up
            if 'is_empty' in metadata and metadata['is_empty']:
                data = pd.read_hdf(path, entity_key.path, where=filter_terms)
                data = data.set_index(list(data.columns))  # undoing transform performed on write
            else:
                data = pd.read_hdf(path, entity_key.path, where=filter_terms, columns=column_filters)

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
    # Our data is indexed, sometimes with no other columns. This leaves an empty dataframe that
    # store.put will silently fail to write in table format.
    if data.empty:
        _write_empty_dataframe(path, entity_key, data)
    else:
        entity_path = entity_key.path
        metadata = {'is_empty': False}
        with pd.HDFStore(path, complevel=9) as store:
            store.put(entity_path, data, format="table")
            store.get_storer(entity_path).attrs.metadata = metadata  # NOTE: must use attrs. write this up


def _write_empty_dataframe(path: str, entity_key: 'EntityKey', data: pd.DataFrame):
    """Writes an empty pandas DataFrame to the hdf file at the given path, queryable by its index."""
    entity_path = entity_key.path
    data = data.reset_index()

    if data.empty:
        raise ValueError("Cannot write an empty dataframe that does not have an index.")

    metadata = {'is_empty': True}
    with pd.HDFStore(path, complevel=9) as store:
        store.put(entity_path, data, format='table', data_columns=True)
        store.get_storer(entity_path).attrs.metadata = metadata  # NOTE: must use attrs. write this up


def _get_keys(root: tables.node.Node, prefix: str = '') -> List[str]:
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
