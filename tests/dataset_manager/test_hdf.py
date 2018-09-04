import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import pytest
import tables
from tables.nodes import filenode
from vivarium.testing_utilities import build_table

from vivarium_public_health.dataset_manager import EntityKey, hdf


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


@pytest.fixture
def hdf_file(hdf_file_path):
    with tables.open_file(hdf_file_path, mode='r') as file:
        yield file


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


@pytest.fixture(params=['totally.new.thing', 'other.new_thing', 'cause.sort_of_new', 'cause.also.new',
                        'cause.all_cause.kind_of_new'])
def mock_key(request):
    return EntityKey(request.param)


@pytest.fixture(params=[[], {}, ['data'], {'thing': 'value'}, 'bananas'])
def json_data(request):
    return request.param


def test_touch_no_file(mocker):
    path = Path('not/an/existing/path.hdf')
    tables_mock = mocker.patch("vivarium_public_health.dataset_manager.hdf.tables")

    hdf.touch(path, False)
    tables_mock.open_file.assert_called_once_with(str(path), mode='w')
    tables_mock.reset_mock()

    with pytest.raises(FileNotFoundError):
        hdf.touch(path, True)


def test_touch_exists_but_not_file(hdf_file_path):
    path = Path(hdf_file_path).parent

    with pytest.raises(FileNotFoundError):
        hdf.touch(path, True)


def test_touch_existing_file(hdf_file_path, mocker):
    path = Path(hdf_file_path)
    tables_mock = mocker.patch("vivarium_public_health.dataset_manager.hdf.tables")

    hdf.touch(path, False)
    tables_mock.open_file.assert_called_once_with(str(path), mode='w')
    tables_mock.reset_mock()

    hdf.touch(path, True)
    tables_mock.open_file.assert_not_called()


def test_write_df(hdf_file_path, mock_key, mocker):
    df_mock = mocker.patch('vivarium_public_health.dataset_manager.hdf._write_data_frame')
    data = pd.DataFrame(np.random.random((10, 3)), columns=['a', 'b', 'c'], index=range(10))

    hdf.write(hdf_file_path, mock_key, data)

    df_mock.assert_called_once_with(hdf_file_path, mock_key, data)


def test_write_json(hdf_file_path, mock_key, json_data, mocker):
    json_mock = mocker.patch('vivarium_public_health.dataset_manager.hdf._write_json_blob')
    hdf.write(hdf_file_path, mock_key, json_data)
    json_mock.assert_called_once_with(hdf_file_path, mock_key, json_data)


def test_load(hdf_file_path, hdf_key):
    key = EntityKey(hdf_key)
    data = hdf.load(hdf_file_path, key, filter_terms=None)
    if 'restrictions' in key:
        assert isinstance(data, dict)
    else:
        assert isinstance(data, pd.DataFrame)


def test_load_with_invalid_filters(hdf_file_path, hdf_key):
    key = EntityKey(hdf_key)
    data = hdf.load(hdf_file_path, key, filter_terms=["fake_filter==0"])
    if 'restrictions' in key:
        assert isinstance(data, dict)
    else:
        assert isinstance(data, pd.DataFrame)


def test_load_with_valid_filters(hdf_file_path, hdf_key):
    key = EntityKey(hdf_key)
    data = hdf.load(hdf_file_path, key, filter_terms=["year == 2006"])
    if 'restrictions' in key:
        assert isinstance(data, dict)
    else:
        assert isinstance(data, pd.DataFrame)
        if 'year' in data.columns:
            assert set(data.year) == {2006}


def test_remove(hdf_file_path, hdf_key):
    key = EntityKey(hdf_key)
    hdf.remove(hdf_file_path, key)
    with tables.open_file(hdf_file_path, mode='r') as file:
        assert key.path not in file


def test_get_keys(hdf_file_path, hdf_keys):
    assert sorted(hdf.get_keys(hdf_file_path)) == sorted(hdf_keys)


def test_write_json_blob(hdf_file_path, mock_key, json_data):
    hdf._write_json_blob(hdf_file_path, mock_key, json_data)

    with tables.open_file(hdf_file_path, mode='r') as file:
        node = file.get_node(mock_key.path)
        with filenode.open_node(node, 'r') as file_node:
            data = json.load(file_node)
            assert data == json_data


def test_write_empty_data_frame(hdf_file_path):
    key = EntityKey('cause.test.prevalence')
    data = pd.DataFrame(columns=('age', 'year', 'sex', 'draw', 'location', 'value'))

    with pytest.raises(ValueError):
        hdf._write_data_frame(hdf_file_path, key, data)


def test_write_data_frame(hdf_file_path):
    key = EntityKey('cause.test.prevalence')
    data = build_table([lambda *args, **kwargs: random.choice([0, 1]), "Kenya", 1],
                       2005, 2010, columns=('age', 'year', 'sex', 'draw', 'location', 'value'))

    hdf._write_data_frame(hdf_file_path, key, data)

    written_data = pd.read_hdf(hdf_file_path, key.path)
    assert written_data.equals(data)

    filter_terms = ['draw == 0']
    written_data = pd.read_hdf(hdf_file_path, key.path, where=filter_terms)
    assert written_data.equals(data[data['draw'] == 0])


def test_get_keys_private(hdf_file, hdf_keys):
    assert sorted(hdf._get_keys(hdf_file.root)) == sorted(hdf_keys)


def test_get_node_name(hdf_file, hdf_key):
    key = EntityKey(hdf_key)
    assert hdf._get_node_name(hdf_file.get_node(key.path)) == key.measure


def test_get_valid_filter_terms_all_invalid(hdf_key, hdf_file):
    node = hdf_file.get_node(EntityKey(hdf_key).path)
    if not isinstance(node, tables.earray.EArray):
        columns = node.table.colnames
        invalid_filter_terms = _construct_no_valid_filters(columns)
        assert hdf._get_valid_filter_terms(invalid_filter_terms, columns) is None


def test_get_valid_filter_terms_all_valid(hdf_key, hdf_file):
    node = hdf_file.get_node(EntityKey(hdf_key).path)
    if not isinstance(node, tables.earray.EArray):
        columns = node.table.colnames
        valid_filter_terms = _construct_all_valid_filters(columns)
        assert set(hdf._get_valid_filter_terms(valid_filter_terms, columns)) == set(valid_filter_terms)


def test_get_valid_filter_terms_some_valid(hdf_key, hdf_file):
    node = hdf_file.get_node(EntityKey(hdf_key).path)
    if not isinstance(node, tables.earray.EArray):
        columns = node.table.colnames
        invalid_filter_terms = _construct_no_valid_filters(columns)
        valid_filter_terms = _construct_all_valid_filters(columns)
        all_terms = invalid_filter_terms + valid_filter_terms
        result = hdf._get_valid_filter_terms(all_terms, columns)
        assert set(result) == set(valid_filter_terms)


def test_get_valid_filter_terms_no_terms():
    assert hdf._get_valid_filter_terms(None, []) is None


def _construct_no_valid_filters(columns):
    fake_cols = [c[1:] for c in columns] # strip out the first char to make a list of all fake cols
    terms = [c + ' <= 0' for c in fake_cols]
    return _complicate_terms_to_parse(terms)


def _construct_all_valid_filters(columns):
    terms = [c + '=0' for c in columns] # assume c is numeric - we won't actually apply filter
    return _complicate_terms_to_parse(terms)


def _complicate_terms_to_parse(terms):
    n_terms = len(terms)
    if n_terms > 1:
        # throw in some parens and ifs/ands
        term_1 = '(' + ' & '.join(terms[:(n_terms//2+n_terms % 2)]) + ')'
        term_2 = '(' + ' | '.join(terms[(n_terms//2+n_terms % 2):]) + ')'
        terms = [term_1, term_2] + terms
    return ['(' + t + ')' for t in terms]
