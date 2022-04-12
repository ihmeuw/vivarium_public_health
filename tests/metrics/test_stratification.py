from datetime import datetime
from math import floor
from typing import Callable, List, Set
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from vivarium_public_health.metrics.stratification import (
    ResultsStratifier,
    Source,
    SourceType,
    StratificationLevel,
)

AGE_BINS = pd.DataFrame(
    [
        {"age_start": age, "age_end": age + 5, "age_group_name": f"{age}_to_{age + 5}"}
        for i, age in enumerate(range(10, 65, 5))
    ]
)
DEFAULT_CONFIGS = ["year", "sex", "age"]
START_YEAR = 2020
END_YEAR = 2030
STRATIFICATION_LEVELS = {
    "age": StratificationLevel(
        name="age",
        sources=[Source("age", SourceType.COLUMN)],
        categories={"0", "1", "2", "3", "4"},
        mapper=lambda row: str(floor(row[0])),
    ),
    "sex": StratificationLevel(
        name="l2",
        sources=[Source("sex", SourceType.COLUMN)],
        categories={"Male", "Female"},
    ),
    "multi": StratificationLevel(
        name="multi",
        sources=[
            Source("part_1", SourceType.COLUMN),
            Source("pipeline_1", SourceType.PIPELINE),
        ],
        categories={"c2", "c30", "c31", "c10", "c11"},
        mapper=lambda row: "c2"
        if row["part_1"] == "c2"
        else row["part_1"] + row["pipeline_1"],
    ),
    "p2": StratificationLevel(
        name="p2",
        sources=[Source("pipeline_2", SourceType.PIPELINE)],
        categories={"2", "3"},
    ),
    "year": StratificationLevel(
        name="year",
        sources=[Source("clock_1", SourceType.CLOCK)],
        categories={"2020", "2021", "2022", "2023"},
        mapper=lambda row: str(row[0].year),
        current_categories_getter=lambda: {str(mock_clock().year)},
    ),
    "month": StratificationLevel(
        name="month",
        sources=[Source("clock_2", SourceType.CLOCK)],
        categories={str(i + 1) for i in range(12)},
        mapper=lambda row: str(row[0].month),
        current_categories_getter=lambda: {str(mock_clock().month)},
    ),
}


def unmock_instance_method(instance, method: Callable) -> Callable:
    return lambda *args: method(instance, *args)


@pytest.fixture
def mock_stratification_level_init(mocker):
    return mocker.patch(
        "vivarium_public_health.metrics.stratification.StratificationLevel.__init__",
        return_value=None,
    )


@pytest.fixture()
def mock_stratifier(mocker) -> MagicMock:
    stratifier = mocker.MagicMock()
    stratifier.setup = mocker.MagicMock(
        side_effect=unmock_instance_method(stratifier, ResultsStratifier.setup)
    )
    return stratifier


@pytest.fixture()
def mock_builder(mocker) -> MagicMock:
    builder = mocker.MagicMock()

    builder.configuration.observers.default = DEFAULT_CONFIGS
    builder.configuration.time.start.year = START_YEAR
    builder.configuration.time.end.year = END_YEAR

    builder.data.load = mocker.MagicMock(side_effect=load)
    builder.value.get_value = mocker.MagicMock(side_effect=get_value)

    builder.time.clock.return_value = mock_clock
    return builder


def mock_clock() -> datetime:
    return datetime(2022, 5, 3, 12)


def load(key: str) -> pd.DataFrame:
    return {
        "population.age_bins": pd.DataFrame(
            [
                {
                    "age_start": age,
                    "age_end": age + 5,
                    "age_group_name": f"{age}_to_{age + 5}",
                }
                for i, age in enumerate(range(0, 100, 5))
            ]
        )
    }[key]


def get_value(pipeline_name: str) -> str:
    return f"{pipeline_name}_pipeline"


def mock_mapper(row: pd.Series) -> str:
    return str(row[0]) + str(row[0])


def mock_current_categories_getter() -> Set[str]:
    return {"a", "b", "c"}


def test_setting_default_stratification(mock_stratifier: MagicMock, mock_builder: MagicMock):
    # Setup mocks
    mock_stratifier._get_default_stratification_levels = MagicMock(
        side_effect=unmock_instance_method(
            mock_stratifier, ResultsStratifier._get_default_stratification_levels
        )
    )

    # Code to test
    mock_stratifier.setup(mock_builder)

    # Assertions
    assert mock_stratifier.default_stratification_levels == set(DEFAULT_CONFIGS)


@pytest.mark.parametrize(
    "age_start, age_end, expected_age_start, expected_age_end",
    [(20, 55, 20, 55), (18, 47, 15, 50)],
    ids=["on_bound", "between_bounds"],
)
def test_setting_age_bins(
    mock_stratifier: MagicMock,
    mock_builder: MagicMock,
    age_start: int,
    age_end: int,
    expected_age_start: int,
    expected_age_end: int,
):
    # Setup mocks
    mock_builder.configuration.population.age_start = age_start
    mock_builder.configuration.population.exit_age = age_end

    mock_stratifier._get_age_bins = MagicMock(
        side_effect=unmock_instance_method(mock_stratifier, ResultsStratifier._get_age_bins)
    )

    # Code to test
    mock_stratifier.setup(mock_builder)

    # Assertions
    expected_outputs = pd.DataFrame(
        [
            {"age_start": age, "age_end": age + 5, "age_group_name": f"{age}_to_{age + 5}"}
            for i, age in enumerate(range(expected_age_start, expected_age_end, 5))
        ]
    )
    assert (mock_stratifier.age_bins.values == expected_outputs.values).all()


def test_stratifications_and_listeners_registered(
    mock_stratifier: MagicMock,
    mock_builder: MagicMock,
):
    # Setup mocks
    mock_stratifier._register_timestep_prepare_listener = MagicMock(
        side_effect=unmock_instance_method(
            mock_stratifier, ResultsStratifier._register_timestep_prepare_listener
        )
    )

    # Code to test
    mock_stratifier.setup(mock_builder)

    # Assertions
    mock_stratifier.register_stratifications.assert_called_once_with(mock_builder)
    mock_builder.event.register_listener.assert_called_once_with(
        "time_step__prepare", mock_stratifier.on_time_step_prepare, priority=0
    )


def test_registering_stratifications(mock_stratifier: MagicMock, mock_builder: MagicMock):
    # Setup mocks
    mock_stratifier.age_bins = AGE_BINS
    mock_stratifier.register_stratifications = MagicMock(
        side_effect=unmock_instance_method(
            mock_stratifier, ResultsStratifier.register_stratifications
        )
    )

    # Code to test
    mock_stratifier.register_stratifications(mock_builder)

    # Assertions
    mock_stratifier.setup_stratification.assert_any_call(
        mock_builder,
        name="year",
        sources=[Source("year", SourceType.CLOCK)],
        categories={str(year) for year in range(START_YEAR, END_YEAR + 1)},
        mapper=mock_stratifier.year_stratification_mapper,
        current_category_getter=mock_stratifier.year_current_categories_getter,
    )

    mock_stratifier.setup_stratification.assert_any_call(
        mock_builder,
        name="sex",
        sources=[Source("sex", SourceType.COLUMN)],
        categories={"Female", "Male"},
    )

    mock_stratifier.setup_stratification.assert_any_call(
        mock_builder,
        name="age",
        sources=[Source("age", SourceType.COLUMN)],
        categories={age_bin for age_bin in AGE_BINS["age_group_name"]},
        mapper=mock_stratifier.age_stratification_mapper,
    )


@pytest.mark.parametrize(
    "sources",
    [
        [Source("age", SourceType.COLUMN)],
        [Source("exposure", SourceType.PIPELINE)],
        [Source("year", SourceType.CLOCK)],
        [Source("age", SourceType.COLUMN), Source("exposure", SourceType.PIPELINE)],
    ],
    ids=["column", "pipeline", "clock", "multiple"],
)
def test_setup_stratification(
    mock_stratifier: MagicMock,
    mock_builder: MagicMock,
    mock_stratification_level_init: patch,
    sources: List[Source],
):
    # Setup mocks
    mock_stratifier.stratification_levels = {}
    mock_stratifier.pipelines = {}
    mock_stratifier.columns_required = {"tracked"}
    mock_stratifier.clock_sources = set()

    mock_stratifier.setup_stratification = MagicMock(
        side_effect=unmock_instance_method(
            mock_stratifier, ResultsStratifier.setup_stratification
        )
    )
    name = "test_name"
    categories = {"a", "b", "c"}

    # Code to test
    mock_stratifier.setup_stratification(
        mock_builder,
        name,
        sources,
        categories,
        mock_mapper,
        mock_current_categories_getter,
    )

    # Assertions
    mock_stratification_level_init.assert_called_once_with(
        name, sources, categories, mock_mapper, mock_current_categories_getter
    )
    assert name in mock_stratifier.stratification_levels

    for source in sources:
        if source.type == SourceType.PIPELINE:
            assert source.name in mock_stratifier.pipelines
            assert mock_stratifier.pipelines[source.name] == get_value(source.name)
        if source.type == SourceType.COLUMN:
            assert source.name in mock_stratifier.columns_required
        if source.type == SourceType.CLOCK:
            assert source.name in mock_stratifier.clock_sources


def test_setting_stratification_groups_on_time_step_prepare(
    mock_stratifier: MagicMock, mock_builder: MagicMock
):
    # Setup mocks
    def mock_pipeline_1(index: pd.Index) -> pd.Series:
        return pd.Series([str(i % 2) for i in range(index.size)], index=index)

    def mock_pipeline_2(index: pd.Index) -> pd.Series:
        return pd.Series([str(i % 2 + 2) for i in range(index.size)], index=index)

    mock_pipelines = {
        "pipeline_1": mock_pipeline_1,
        "pipeline_2": mock_pipeline_2,
    }

    mock_population_view_data = pd.DataFrame(
        {
            "age": [4.5, 3.4, 0.2, 0.3],
            "sex": ["Male", "Female", "Female", "Male"],
            "part_1": ["c2", "c2", "c3", "c1"],
        }
    )

    mock_clock_sources = {"clock_1", "clock_2"}

    mock_event = MagicMock()
    mock_event.index = mock_population_view_data.index

    mock_stratifier.population_view.get.return_value = mock_population_view_data
    mock_stratifier.pipelines = mock_pipelines
    mock_stratifier.clock_sources = mock_clock_sources
    mock_stratifier.clock = mock_clock
    mock_stratifier.stratification_levels = STRATIFICATION_LEVELS

    mock_stratifier.on_time_step_prepare = MagicMock(
        side_effect=unmock_instance_method(
            mock_stratifier, ResultsStratifier.on_time_step_prepare
        )
    )

    # Code to test
    mock_stratifier.on_time_step_prepare(mock_event)

    # Assertions
    expected_groups = pd.DataFrame(
        {
            "age": ["4", "3", "0", "0"],
            "sex": ["Male", "Female", "Female", "Male"],
            "multi": ["c2", "c2", "c30", "c11"],
            "p2": ["2", "3", "2", "3"],
            "year": ["2022", "2022", "2022", "2022"],
            "month": ["5", "5", "5", "5"],
        },
        index=mock_population_view_data.index,
    )
    assert (mock_stratifier.stratification_groups.values == expected_groups.values).all()


# todo get test_group to work
# TEST_GROUP_INDEX = pd.Index(list(range(4)))
#
#
# @pytest.mark.parametrize(
#     "include, exclude, expected",
#     [
#         ([], DEFAULT_CONFIGS, {"": True}),
#         ([], ["age", "year"], {
#             "sex_male": pd.Series([True, False, False, True], index=TEST_GROUP_INDEX),
#             "sex_female": pd.Series([False, True, True, False], index=TEST_GROUP_INDEX)
#         }),
#     ],
#     ids=[
#         'no stratification',
#         "sex_only",
#     ]
# )
# def test_group(
#         mock_stratifier: MagicMock,
#         include: Set[str],
#         exclude: Set[str],
#         expected: Dict[str, pd.Series]
# ):
#     # Setup mocks
#     mock_stratifier.default_stratification_levels = set(DEFAULT_CONFIGS)
#     mock_stratifier.stratification_levels = STRATIFICATION_LEVELS
#     mock_stratifier.stratification_groups = pd.DataFrame(
#         {
#             "age": ["4", "3", "0", "0", "3"],
#             "sex": ["Male", "Female", "Female", "Male", "Male"],
#             "multi": ["c2", "c2", "c30", "c11", "c11"],
#             "p2": ["2", "3", "2", "3", "3"],
#             "year": ["2022", "2022", "2022", "2022", "2022"],
#             "month": ["5", "5", "5", "5", "5"],
#         }
#     )
#
#     mock_stratifier.group = MagicMock(
#         side_effect=unmock_instance_method(mock_stratifier, ResultsStratifier.group)
#     )
#     mock_stratifier._get_current_stratifications = MagicMock(
#         side_effect=unmock_instance_method(
#             mock_stratifier, ResultsStratifier._get_current_stratifications
#         )
#     )
#     mock_stratifier._get_stratification_key = MagicMock(
#         side_effect=ResultsStratifier._get_stratification_key
#     )
#
#     # Code to test
#     groups = mock_stratifier.group(TEST_GROUP_INDEX, include, exclude)
#     number_of_groups = 0
#
#     # Assertions
#     for label, group_mask in groups:
#         assert (
#             group_mask == expected[label] if isinstance(group_mask, bool)
#             else (group_mask == expected[label]).all()
#         )
#         number_of_groups += 1
#
#     assert number_of_groups == len(expected)

# todo test age_stratification_mapper
# todo test year_stratification_mapper
# todo test year_current_categories_getter
# todo test StratificationLevel init
