import itertools
from datetime import datetime
from math import floor
from typing import Callable, List, Set
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from vivarium.testing_utilities import TestPopulation, metadata
from vivarium import InteractiveContext
from vivarium.framework.engine import Builder
from vivarium.framework.population import SimulantData

from vivarium_public_health.metrics.stratification import (
    ResultsStratifier as ResultsStratifier_,
    Source,
    SourceType,
    StratificationLevel,
)


class FavoriteColor:

    OPTIONS = ['red', 'green', 'orange']

    @property
    def name(self) -> str:
        return 'favorite_color'

    def setup(self, builder: Builder):
        self.randomness = builder.randomness.get_stream(self.name)
        # Our simulants are fickle and change their
        # favorite colors every time step.
        builder.value.register_value_producer(
            self.name,
            requires_streams=[self.name],
            source=lambda index: self.randomness.choice(
                index, choices=self.OPTIONS
            )
        )


class FavoriteNumber:

    OPTIONS = [7, 42, 14312]

    @property
    def name(self) -> str:
        return 'favorite_number'

    def setup(self, builder: Builder) -> None:
        self.randomness = builder.randomness.get_stream(self.name)
        self.population_view = builder.population.get_view([self.name])
        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            creates_columns=[self.name],
            requires_streams=[self.name],
        )

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        # Favorite numbers are forever though.
        self.population_view.update(
            self.randomness.choice(
                pop_data.index,
                choices=self.OPTIONS,
            ).rename(self.name)
        )

FAVORITE_THINGS = {f'{color}_{number}' for color, number
                   in itertools.product(FavoriteColor.OPTIONS, FavoriteNumber.OPTIONS)}


class ResultsStratifier(ResultsStratifier_):

    def register_stratifications(self, builder: Builder) -> None:
        super().register_stratifications(builder)

        self.setup_stratification(
            builder,
            name="favorite_things",
            sources=[
                Source('favorite_color', SourceType.PIPELINE),
                Source('favorite_number', SourceType.COLUMN),
            ],
            categories=FAVORITE_THINGS,
            mapper=lambda row: f"{row['favorite_color']}_{row['favorite_number']}",
        )

        self.setup_stratification(
            builder,
            name="favorite_color",
            sources=[Source("favorite_color", SourceType.PIPELINE)],
            categories=set(FavoriteColor.OPTIONS),
        )

        self.setup_stratification(
            builder,
            name="month",
            sources=[Source("month", SourceType.CLOCK)],
            categories={str(i + 1) for i in range(12)},
            mapper=lambda row: str(row[0].month),
            current_category_getter=lambda: {str(self.clock().month)},
        )


@pytest.fixture(params=[
    [],
    ['age', 'sex', 'year'],
    ['age', 'sex', 'year'] + ['favorite_things', 'favorite_color', 'month'],
])
def stratification_levels(request):
    return request.param


@pytest.fixture(scope='function')
def stratifier_and_sim(stratification_levels, base_config, base_plugins):
    base_config.update(
        {'observers': {'default': stratification_levels},
         'population': {'population_size': 1000}},
        **metadata(__file__)
    )
    rs = ResultsStratifier()
    sim = InteractiveContext(
        components=[TestPopulation(), rs, FavoriteColor(), FavoriteNumber()],
        configuration=base_config,
        plugin_configuration=base_plugins,
    )
    return rs, sim


def test_ResultsStratifier_setup(stratifier_and_sim, stratification_levels):
    rs, sim = stratifier_and_sim

    assert rs.metrics_pipeline_name == 'metrics'
    assert rs.tmrle_key == "population.theoretical_minimum_risk_life_expectancy"
    assert rs.clock() == sim.current_time
    assert rs.default_stratification_levels == set(stratification_levels)
    assert set(rs.pipelines) == {'favorite_color'}
    assert rs.columns_required == {'tracked', 'favorite_number', 'age', 'sex'}
    assert rs.clock_sources == {'year', 'month'}
    assert set(rs.stratification_levels) == {'age', 'sex', 'year', 'favorite_things',
                                             'favorite_color', 'month'}
    assert rs.stratification_groups is None


@pytest.mark.parametrize(
    "age_start, age_end, expected_age_start, expected_age_end",
    [(20, 55, 20, 55), (18, 47, 15, 50)],
    ids=["on_bound", "between_bounds"],
)
def test_ResultsStratifier_setup_age_bins(
    age_start: int,
    age_end: int,
    expected_age_start: int,
    expected_age_end: int,
    base_config, base_plugins,
):
    base_config.update(
        {'population': {'age_start': age_start, 'exit_age': age_end}},
        **metadata(__file__),
    )

    rs = ResultsStratifier()
    _ = InteractiveContext(
        components=[TestPopulation(), rs],
        configuration=base_config,
        plugin_configuration=base_plugins,
    )

    # Assertions
    expected_outputs = pd.DataFrame(
        [
            {"age_start": float(age),
             "age_end": float(age + 5),
             "age_group_name": f"{age} to {age + 4}"}
            for i, age in enumerate(range(expected_age_start, expected_age_end, 5))
        ]
    )
    assert (rs.age_bins.values == expected_outputs.values).all()


def test_setting_stratification_groups_on_time_step_prepare(stratifier_and_sim):
    rs, sim = stratifier_and_sim

    # No time steps yet, so not groups have been set
    assert rs.stratification_groups is None

    stratification_time = sim.current_time
    sim.step()
    # noinspection PyTypeChecker
    sg: pd.DataFrame = rs.stratification_groups

    def proportion(x):
        return len(x) / len(sg)

    assert set(sg.columns) == {'age', 'sex', 'year', 'favorite_things',
                               'favorite_color', 'month'}

    # Age has different sized bins, so statistical tests are harder.
    assert set(sg.age) <= set(rs.age_bins.age_group_name)

    assert set(sg.sex) == {'Male', 'Female'}
    assert np.allclose(
        sg.groupby('sex').apply(proportion),
        1 / 2,
        rtol=0.1
    )

    years = set(sg.year)
    assert len(years) == 1
    assert list(years)[0] == str(stratification_time.year)

    assert set(sg.favorite_things) == FAVORITE_THINGS
    assert np.allclose(
        sg.groupby('favorite_things').apply(proportion),
        1 / len(FAVORITE_THINGS),
        rtol=0.25,
    )

    assert set(sg.favorite_color) == set(FavoriteColor.OPTIONS)
    assert np.allclose(
        sg.groupby('favorite_color').apply(proportion),
        1 / len(FavoriteColor.OPTIONS),
        rtol=0.1
    )

    months = set(sg.month)
    assert len(months) == 1
    assert list(months)[0] == str(stratification_time.month)


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
