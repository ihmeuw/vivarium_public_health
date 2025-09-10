import math
from itertools import product

import numpy as np
import pandas as pd
import pytest
from vivarium import InteractiveContext
from vivarium.testing_utilities import get_randomness
from vivarium_testing_utils import FuzzyChecker

import vivarium_public_health.population.base_population as bp
import vivarium_public_health.population.data_transformations as dt
from tests.mock_artifact import MockArtifact
from tests.test_utilities import make_uniform_pop_data, simple_pop_structure
from vivarium_public_health import utilities


def test_select_sub_population_data():
    data = pd.DataFrame(
        {
            "year_start": [1990, 1995, 2000, 2005],
            "year_end": [1995, 2000, 2005, 2010],
            "population": [100, 110, 120, 130],
        }
    )

    sub_pop = bp.BasePopulation.get_demographic_proportions_for_creation_time(data, 1997)

    assert sub_pop.year_start.values.item() == 1995


def test_BasePopulation(
    config, full_simulants, base_plugins, generate_population_mock, include_sex
):
    num_days = 600
    time_step = 100  # Days
    start_population_size = len(full_simulants)

    generate_population_mock.return_value = full_simulants.drop(columns=["tracked"])

    base_pop = bp.BasePopulation()

    components = [base_pop]
    config.update(
        {
            "population": {
                "population_size": start_population_size,
                "include_sex": include_sex,
            },
            "time": {"step_size": time_step},
        },
        layer="override",
    )
    simulation = InteractiveContext(
        components=components, configuration=config, plugin_configuration=base_plugins
    )
    time_start = simulation._clock.time

    pop_structure = simulation._data.load("population.structure")
    uniform_pop = dt.assign_demographic_proportions(pop_structure, include_sex)

    assert base_pop.demographic_proportions.equals(uniform_pop)

    age_params = {
        "age_start": config.population.initialization_age_min,
        "age_end": config.population.initialization_age_max,
    }
    sub_pop = bp.BasePopulation.get_demographic_proportions_for_creation_time(
        uniform_pop, time_start.year
    )

    generate_population_mock.assert_called_once()
    # Get a dictionary of the arguments used in the call
    mock_args = generate_population_mock.call_args[1]
    assert mock_args["creation_time"] == time_start - simulation._clock.step_size
    assert mock_args["age_params"] == age_params
    assert mock_args["demographic_proportions"].equals(sub_pop)
    assert mock_args["randomness_streams"] == base_pop.randomness
    pop = simulation.get_population()
    for column in pop:
        assert pop[column].equals(full_simulants[column])

    final_ages = pop.age + num_days / utilities.DAYS_PER_YEAR

    simulation.run_for(duration=pd.Timedelta(days=num_days))

    pop = simulation.get_population()
    assert np.allclose(
        pop.age, final_ages, atol=0.5 / utilities.DAYS_PER_YEAR
    )  # Within a half of a day.


def test_age_out_simulants(config, base_plugins):
    start_population_size = 10000
    num_days = 600
    time_step = 100  # Days
    config.update(
        {
            "population": {
                "population_size": start_population_size,
                "initialization_age_min": 4,
                "initialization_age_max": 4,
                "untracking_age": 5,
            },
            "time": {"step_size": time_step},
        },
        layer="override",
    )
    components = [bp.BasePopulation()]
    simulation = InteractiveContext(
        components=components, configuration=config, plugin_configuration=base_plugins
    )
    time_start = simulation._clock.time
    assert len(simulation.get_population()) == len(simulation.get_population().age.unique())
    simulation.run_for(duration=pd.Timedelta(days=num_days))
    pop = simulation.get_population()
    assert len(pop) == len(pop[~pop.tracked])
    exit_after_300_days = pop.exit_time >= time_start + pd.Timedelta(300, unit="D")
    exit_before_400_days = pop.exit_time <= time_start + pd.Timedelta(400, unit="D")
    assert len(pop) == len(pop[exit_after_300_days & exit_before_400_days])


def test_generate_population_age_bounds(
    base_simulants, age_bounds_mock, initial_age_mock, include_sex
):
    creation_time = pd.Timestamp(1990, 7, 2)
    step_size = pd.Timedelta(days=1)
    age_params = {"age_start": 0, "age_end": 120}
    pop_data = dt.assign_demographic_proportions(
        make_uniform_pop_data(age_bin_midpoint=True),
        include_sex=include_sex,
    )
    r = {k: get_randomness() for k in ["general_purpose", "bin_selection", "age_smoothing"]}
    simulant_ids = base_simulants.index

    bp.generate_population(
        simulant_ids,
        creation_time,
        step_size,
        age_params,
        pop_data,
        r,
        lambda *args, **kwargs: None,
    )

    age_bounds_mock.assert_called_once()
    mock_args = age_bounds_mock.call_args[0]
    assert mock_args[0].equals(base_simulants)
    assert mock_args[1].equals(pop_data)
    assert mock_args[2] == float(age_params["age_start"])
    assert mock_args[3] == float(age_params["age_end"])
    assert mock_args[4] == r
    initial_age_mock.assert_not_called()


def test_generate_population_initial_age(
    base_simulants, age_bounds_mock, initial_age_mock, include_sex
):
    creation_time = pd.Timestamp(1990, 7, 2)
    step_size = pd.Timedelta(days=1)
    age_params = {"age_start": 0, "age_end": 0}
    pop_data = dt.assign_demographic_proportions(
        make_uniform_pop_data(age_bin_midpoint=True),
        include_sex=include_sex,
    )
    r = {k: get_randomness() for k in ["general_purpose", "bin_selection", "age_smoothing"]}
    simulant_ids = base_simulants.index

    bp.generate_population(
        simulant_ids,
        creation_time,
        step_size,
        age_params,
        pop_data,
        r,
        lambda *args, **kwargs: None,
    )

    initial_age_mock.assert_called_once()
    mock_args = initial_age_mock.call_args[0]
    assert mock_args[0].equals(base_simulants)
    assert mock_args[1].equals(pop_data)

    assert mock_args[2] == float(age_params["age_start"])
    assert mock_args[3] == step_size
    assert mock_args[4] == r
    age_bounds_mock.assert_not_called()


def test__assign_demography_with_initial_age(config, base_simulants, include_sex):
    pop_data = dt.assign_demographic_proportions(
        make_uniform_pop_data(age_bin_midpoint=True),
        include_sex=include_sex,
    )
    pop_data = pop_data[pop_data.year_start == 1990]
    initial_age = 20
    r = {k: get_randomness() for k in ["general_purpose", "bin_selection", "age_smoothing"]}
    step_size = pd.Timedelta(days=config.time.step_size)

    base_simulants = bp._assign_demography_with_initial_age(
        base_simulants, pop_data, initial_age, step_size, r, lambda *args, **kwargs: None
    )
    _check_population(base_simulants, initial_age, step_size, include_sex)


def test__assign_demography_with_initial_age_zero(base_simulants, config, include_sex):
    pop_data = dt.assign_demographic_proportions(
        make_uniform_pop_data(age_bin_midpoint=True),
        include_sex=include_sex,
    )
    pop_data = pop_data[pop_data.year_start == 1990]
    initial_age = 0
    r = {k: get_randomness() for k in ["general_purpose", "bin_selection", "age_smoothing"]}
    step_size = utilities.to_time_delta(config.time.step_size)

    base_simulants = bp._assign_demography_with_initial_age(
        base_simulants, pop_data, initial_age, step_size, r, lambda *args, **kwargs: None
    )
    _check_population(base_simulants, initial_age, step_size, include_sex)


def test__assign_demography_with_initial_age_error(base_simulants, include_sex):
    pop_data = dt.assign_demographic_proportions(
        make_uniform_pop_data(age_bin_midpoint=True),
        include_sex=include_sex,
    )
    pop_data = pop_data[pop_data.year_start == 1990]
    initial_age = 200
    r = {k: get_randomness() for k in ["general_purpose", "bin_selection", "age_smoothing"]}
    step_size = pd.Timedelta(days=1)

    with pytest.raises(ValueError):
        bp._assign_demography_with_initial_age(
            base_simulants, pop_data, initial_age, step_size, r, lambda *args, **kwargs: None
        )


@pytest.mark.parametrize(["age_start", "age_end"], [[0, 180], [5, 50], [12, 57]])
def test__assign_demography_with_age_bounds(base_simulants, include_sex, age_start, age_end):
    pop_data = dt.assign_demographic_proportions(
        make_uniform_pop_data(age_bin_midpoint=True),
        include_sex=include_sex,
    )
    pop_data = pop_data[pop_data.year_start == 1990]
    r = {
        k: get_randomness(k)
        for k in [
            "general_purpose",
            "bin_selection",
            "age_smoothing",
            "age_smoothing_age_bounds",
        ]
    }

    base_simulants = bp._assign_demography_with_age_bounds(
        base_simulants, pop_data, age_start, age_end, r, lambda *args, **kwargs: None
    )

    _check_sexes(base_simulants, include_sex)
    _check_locations(base_simulants)

    ages = np.sort(base_simulants.age.values)
    age_deltas = ages[1:] - ages[:-1]

    age_start = max(pop_data.age_start.min(), age_start)
    age_end = min(pop_data.age_end.max(), age_end)
    expected_average_delta = (age_end - age_start) / len(base_simulants)

    assert math.isclose(age_deltas.mean(), expected_average_delta, rel_tol=1e-2)
    # Make sure there are no big age gaps.
    assert age_deltas.max() < 100 * expected_average_delta


def test__assign_demography_with_age_bounds_error(base_simulants, include_sex):
    pop_data = dt.assign_demographic_proportions(
        make_uniform_pop_data(age_bin_midpoint=True),
        include_sex=include_sex,
    )
    age_start, age_end = 130, 140
    r = {k: get_randomness() for k in ["general_purpose", "bin_selection", "age_smoothing"]}

    with pytest.raises(ValueError):
        bp._assign_demography_with_age_bounds(
            base_simulants, pop_data, age_start, age_end, r, lambda *args, **kwargs: None
        )


@pytest.mark.parametrize("constructor_type", ["string", "data"])
def test_scaled_population(
    constructor_type, config, base_plugins, mocker, fuzzy_checker: FuzzyChecker
):
    config.update(
        {
            "population": {
                "population_size": 1_000_000,
                "include_sex": "Both",
            },
            "time": {
                "step_size": 1,
                # Update the start year to fall within the population structure data.
                "start": {"year": 2021},
            },
        },
        layer="override",
    )

    # Simple pop data
    pop_structure = simple_pop_structure()
    # Simple scalar data to pass to ScaledPopulation
    scalar_data = simple_pop_structure().drop(columns=["location"])
    scalar_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    scalar_data["value"] = scalar_values

    # Add data to artifact and mock return for plugin
    mock_art = MockArtifact()
    mock_art.write("population.structure", pop_structure)
    mocker.patch(
        "tests.mock_artifact.MockArtifactManager._load_artifact",
    ).return_value = mock_art

    if constructor_type == "string":
        mock_art.write("population.scalar", scalar_data)
        scaling_factor = "population.scalar"
    else:
        scaling_factor = scalar_data
    scaled_pop = bp.ScaledPopulation(scaling_factor)
    sim = InteractiveContext(
        components=[scaled_pop], configuration=config, plugin_configuration=base_plugins
    )
    pop = sim.get_population()
    # Use FuzzyChecker to compare population structure to demographic proportion by
    # iterating through each age_group/sex combination
    scaled_structure = pop_structure.copy()
    scaled_structure["value"] = scaled_structure["value"] * scalar_data["value"]

    for row in range(len(scaled_structure)):
        row_data = scaled_structure.iloc[row]
        age_start = row_data["age_start"]
        sex = row_data["sex"]
        # Get proportion of each age group
        target_proportion = row_data["value"] / scaled_structure["value"].sum()
        number_of_sims = len(
            pop.loc[
                (pop["age"] >= age_start)
                & (pop["age"] <= row_data["age_end"])
                & (pop["sex"] == sex)
            ]
        )
        fuzzy_checker.fuzzy_assert_proportion(
            observed_numerator=number_of_sims,
            observed_denominator=len(pop),
            target_proportion=target_proportion,
            name=f"scaled_pop_proportion_check_{sex}_{age_start}",
        )


@pytest.mark.parametrize("test_case", ["pop_structure", "both"])
@pytest.mark.parametrize("year", [2021, 2025])
def test_scaled_population__format_data_inputs(
    test_case: str,
    year: int,
) -> None:
    """Test ScaledPopulation with multi-year data in population structure and/or scaling factor.

    Cases:
    1. pop_structure: Population structure has multiple years, scaling factor does not
    2. both: Both population structure and scaling factor have multiple years
    """

    # Create multi-year population structure (2021-2024)
    age_idx = pd.MultiIndex.from_tuples(
        [
            (0.0, 25.0, "Young People"),
            (25.0, 50.0, "Old People"),
            (50.0, 75.0, "Ancient People"),
            (75.0, 100.0, "People Who Beat the Odds"),
        ],
        names=["age_start", "age_end", "age_group_name"],
    )
    age_df = pd.DataFrame(index=age_idx).reset_index()
    age_bins = [(group.age_start, group.age_end) for group in age_df.itertuples()]
    sexes = ("Male", "Female")
    location = ["Kenya"]
    years = [(2021, 2022), (2022, 2023), (2023, 2024)]  # Multiple years

    age_bins, sexes, years, location = zip(*product(age_bins, sexes, years, location))
    mins, maxes = zip(*age_bins)
    year_starts, year_ends = zip(*years)

    pop_structure = pd.DataFrame(
        {
            "location": location,
            "sex": sexes,
            "age_start": mins,
            "age_end": maxes,
            "year_start": year_starts,
            "year_end": year_ends,
            # Base values for each demographic group
            "value": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0] * 3,  # Repeated for 3 years
        }
    )

    # Create scaling factor data based on test case
    if test_case == "pop_structure":
        # Scaling factor single year
        scalar_data = pd.DataFrame(
            {
                "sex": ["Male", "Female"] * 4,
                "age_start": [0.0, 0.0, 25.0, 25.0, 50.0, 50.0, 75.0, 75.0],
                "age_end": [25.0, 25.0, 50.0, 50.0, 75.0, 75.0, 100.0, 100.0],
                "year_start": [2021] * 8,
                "year_end": [2022] * 8,
                "value": [1.5, 1.2, 2.1, 1.8, 0.9, 1.1, 1.3, 1.4],
            }
        )
        # Remove year_start and year end since scalar data only has one year
        formatted_scalar_data = scalar_data.set_index(
            ["sex", "age_start", "age_end"], drop=True
        )[["value"]]
    elif test_case == "both":
        # Both have multiple years - should be able to multiply together directly
        scalar_data = pd.DataFrame(
            {
                "age_start": [0.0, 0.0, 25.0, 25.0, 50.0, 50.0, 75.0, 75.0] * 3,
                "age_end": [25.0, 25.0, 50.0, 50.0, 75.0, 75.0, 100.0, 100.0] * 3,
                "sex": ["Male", "Female"] * 12,
                "year_start": [2021] * 8 + [2022] * 8 + [2023] * 8,
                "year_end": [2022] * 8 + [2023] * 8 + [2024] * 8,
                "value": [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.5,
                    1.2,
                    2.1,
                    1.8,
                    0.9,
                    1.1,
                    1.3,
                    1.4,
                    1.8,
                    1.4,
                    2.3,
                    2.0,
                    1.2,
                    1.6,
                    1.7,
                    1.9,
                ],
            }
        )

        # Subset to 2021 or subset to 2023 since that is latest year start to verify
        # we are subsetting to correct year in _format_data_inputs
        query_year = 2021 if year == 2021 else 2023
        formatted_pop_structure = pop_structure.loc[pop_structure["year_start"] == query_year]
        formatted_scalar_data = scalar_data.loc[scalar_data["year_start"] == query_year]
        formatted_scalar_data = formatted_scalar_data.set_index(
            ["sex", "age_start", "age_end", "year_start", "year_end"], drop=True
        )
        formatted_pop_structure = formatted_pop_structure.set_index(
            ["location", "sex", "age_start", "age_end", "year_start", "year_end"]
        )
        scaled_pop = bp.ScaledPopulation("placeholder")
        formatted = scaled_pop._format_data_inputs(pop_structure, scalar_data, year)
        expected = (formatted[0] * formatted[1]).reset_index()
        data = (formatted_pop_structure * formatted_scalar_data).reset_index()
        pd.testing.assert_frame_equal(data, expected)


def test__find_bin_start_index():
    sorted_values = [10, 20, 30]
    assert bp._find_bin_start_index(10, sorted_values) == 0
    assert bp._find_bin_start_index(19, sorted_values) == 0
    assert bp._find_bin_start_index(20, sorted_values) == 1
    assert bp._find_bin_start_index(29, sorted_values) == 1
    assert bp._find_bin_start_index(30, sorted_values) == 2
    assert bp._find_bin_start_index(99999, sorted_values) == 2

    # Edge case
    with pytest.raises(
        ValueError, match="The provided value 9 is less than the minimum reference value 10."
    ):
        bp._find_bin_start_index(9, sorted_values)


def _check_population(simulants, initial_age, step_size, include_sex):
    assert len(simulants) == len(simulants.age.unique())
    assert simulants.age.min() > initial_age
    assert simulants.age.max() < initial_age + utilities.to_years(step_size)
    _check_sexes(simulants, include_sex)
    _check_locations(simulants)


def _check_sexes(simulants, include_sex):
    male_prob, female_prob = {
        "Male": (1.0, 0.0),
        "Female": (0.0, 1.0),
        "Both": (0.5, 0.5),
    }[include_sex]
    for sex, prob in [("Male", male_prob), ("Female", female_prob)]:
        assert math.isclose(
            len(simulants[simulants.sex == sex]) / len(simulants),
            prob,
            abs_tol=0.01,
        )


def _check_locations(simulants):
    for location in simulants.location.unique():
        assert math.isclose(
            len(simulants[simulants.location == location]) / len(simulants),
            1 / len(simulants.location.unique()),
            abs_tol=0.01,
        )
