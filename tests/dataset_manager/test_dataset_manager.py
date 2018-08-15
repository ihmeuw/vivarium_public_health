import random

import pytest
from vivarium.testing_utilities import build_table

from vivarium_public_health.dataset_manager.dataset_manager import _subset_rows, _subset_columns, _get_location_term


def test_subset_rows_extra_filters():
    data = build_table(1, 1990, 2010)
    with pytest.raises(ValueError):
        _subset_rows(data, missing_thing=12)


def test_subset_rows():
    values = [lambda *args, **kwargs: random.choice(['red', 'blue']),
              lambda *args, **kwargs: random.choice([1, 2, 3])]
    data = build_table(values, 1990, 2010, columns=('age', 'year', 'sex', 'color', 'number'))

    filtered_data = _subset_rows(data, color='red', number=3)
    assert filtered_data.equals(data[(data.color == 'red') & (data.number == 3)])

    filtered_data = _subset_rows(data, color='red', number=[2, 3])
    assert filtered_data.equals(data[(data.color == 'red') & ((data.number == 2) | (data.number == 3))])


def test_subset_columns():
    values = [0, 'Kenya', 12, 35, 'red', 100]
    data = build_table(values, 1990, 2010, columns=('age', 'year', 'sex', 'draw', 'location',
                                                    'age_group_start', 'age_group_end', 'color', 'value'))

    filtered_data = _subset_columns(data, keep_age_group_edges=False)
    assert filtered_data.equals(data[['age', 'year', 'sex', 'color', 'value']])

    filtered_data = _subset_columns(data, keep_age_group_edges=False, color='red')
    assert filtered_data.equals(data[['age', 'year', 'sex', 'value']])

    filtered_data = _subset_columns(data, keep_age_group_edges=True)
    assert filtered_data.equals(data[['age', 'year', 'sex', 'age_group_start', 'age_group_end', 'color', 'value']])

    filtered_data = _subset_columns(data, keep_age_group_edges=True, color='red')
    assert filtered_data.equals(data[['age', 'year', 'sex', 'age_group_start', 'age_group_end', 'value']])


def test_location_term():
    assert _get_location_term("Cote d'Ivoire") == 'location == "Cote d\'Ivoire" | location == "Global"'
    assert _get_location_term("Kenya") == "location == 'Kenya' | location == 'Global'"
    with pytest.raises(NotImplementedError):
        _get_location_term("W'eird \"location\"")
