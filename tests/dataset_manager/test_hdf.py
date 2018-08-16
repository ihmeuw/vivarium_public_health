import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import pytest
import tables
from tables.nodes import filenode
from vivarium.testing_utilities import build_table

from vivarium_public_health.dataset_manager import EntityKey
from vivarium_public_health.dataset_manager.hdf import (write, load, remove, get_keys, _write_json_blob,
                                                        _write_data_frame, _get_keys, _get_node_name)


@pytest.fixture
def hdf_file_path(tmpdir):
    """This file contains the following:

    Object Tree:
        / (RootGroup) ''
        /cause (Group) ''
        /population (Group) ''
        /population/age_bins (Group) ''
        /population/age_bins/table (Table(23,), shuffle, zlib(9)) ''
        /population/structure (Group) ''
        /population/structure/table (Table(1863,), shuffle, zlib(9)) ''
        /population/theoretical_minimum_risk_life_expectancy (Group) ''
        /population/theoretical_minimum_risk_life_expectancy/table (Table(10502,), shuffle, zlib(9)) ''
        /population/structure/meta (Group) ''
        /population/structure/meta/values_block_1 (Group) ''
        /population/structure/meta/values_block_1/meta (Group) ''
        /population/structure/meta/values_block_1/meta/table (Table(3,), shuffle, zlib(9)) ''
        /cause/all_causes (Group) ''
        /cause/all_causes/restrictions (EArray(166,)) ''
    """
    # Make temporary copy of file for test.
    p = tmpdir.join('artifact.hdf')
    with tables.open_file(str(Path(__file__).parent / 'artifact.hdf'), mode='r') as file:
        file.copy_file(str(p), overwrite=True)
    return str(p)


_KEYS = ['population.age_bins',
         'population.structure',
         'population.theoretical_minimum_risk_life_expectancy',
         'cause.all_causes.restrictions']

@pytest.fixture
def hdf_keys():
    return _KEYS


@pytest.fixture(params=_KEYS)
def hdf_key(request):
    return request.param


@pytest.fixture
def hdf_file(hdf_file_path):
    with tables.open_file(hdf_file_path, mode='r') as file:
        yield file


@pytest.fixture(params=['totally.new.thing', 'other.new_thing', 'cause.sort_of_new', 'cause.also.new',
                        'cause.all_cause.kind_of_new'])
def mock_key(request):
    return EntityKey(request.param)


@pytest.fixture(params=[[], {}, ['data'], {'thing': 'value'}, 'bananas'])
def json_data(request):
    return request.param


def test_write_df(hdf_file_path, mock_key, mocker):
    df_mock = mocker.patch('vivarium_public_health.dataset_manager.hdf._write_data_frame')
    data = pd.DataFrame(np.random.random((10, 3)), columns=['a', 'b', 'c'], index=range(10))

    write(hdf_file_path, mock_key, data)

    df_mock.assert_called_once_with(hdf_file_path, mock_key, data)


def test_write_json(hdf_file_path, mock_key, json_data, mocker):
    json_mock = mocker.patch('vivarium_public_health.dataset_manager.hdf._write_json_blob')
    write(hdf_file_path, mock_key, json_data)
    json_mock.assert_called_once_with(hdf_file_path, mock_key, json_data)


def test_load(hdf_file_path, hdf_key):
    key = EntityKey(hdf_key)
    data = load(hdf_file_path, key, filter_terms=None)
    if 'restrictions' in key:
        assert isinstance(data, dict)
    else:
        assert isinstance(data, pd.DataFrame)


def test_remove(hdf_file_path, hdf_key):
    key = EntityKey(hdf_key)
    remove(hdf_file_path, key)
    with tables.open_file(hdf_file_path, mode='r') as file:
        assert key.path not in file


def test_get_keys(hdf_file_path, hdf_keys):
    assert sorted(get_keys(hdf_file_path)) == sorted(hdf_keys)


def test_write_json_blob(hdf_file_path, mock_key, json_data):
    _write_json_blob(hdf_file_path, mock_key, json_data)

    with tables.open_file(hdf_file_path, mode='r') as file:
        node = file.get_node(mock_key.path)
        with filenode.open_node(node, 'r') as file_node:
            data = json.load(file_node)
            assert data == json_data


def test_write_empty_data_frame(hdf_file_path):
    key = EntityKey('cause.test.prevalence')
    data = pd.DataFrame(columns=('age', 'year', 'sex', 'draw', 'location', 'value'))

    with pytest.raises(ValueError):
        _write_data_frame(hdf_file_path, key, data)


def test_write_data_frame(hdf_file_path):
    key = EntityKey('cause.test.prevalence')
    data = build_table([lambda *args, **kwargs: random.choice([0, 1]), "Kenya", 1],
                       2005, 2010, columns=('age', 'year', 'sex', 'draw', 'location', 'value'))

    _write_data_frame(hdf_file_path, key, data)

    written_data = pd.read_hdf(hdf_file_path, key.path)
    assert written_data.equals(data)

    filter_terms = ['draw == 0']
    written_data = pd.read_hdf(hdf_file_path, key.path, where=filter_terms)
    assert written_data.equals(data[data['draw'] == 0])


def test_get_keys_private(hdf_file, hdf_keys):
    assert sorted(_get_keys(hdf_file.root)) == sorted(hdf_keys)


def test_get_node_name(hdf_file, hdf_key):
    key = EntityKey(hdf_key)
    assert _get_node_name(hdf_file.get_node(key.path)) == key.measure
