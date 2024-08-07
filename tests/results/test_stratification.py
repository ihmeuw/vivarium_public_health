import pandas as pd

from vivarium_public_health.results.stratification import ResultsStratifier


# Age bins prior to get_age_bins
def fake_data_load_population_age_bins(*args):
    AGE_BINS_RAW_DICT = {
        "age_start": {0: 0.0, 1: 0.01917808, 2: 0.07671233, 3: 0.5, 4: 1.0, 5: 2.0},
        "age_end": {0: 0.01917808, 1: 0.07671233, 2: 0.5, 3: 1.0, 4: 2.0, 5: 5.0},
        "age_group_name": {
            0: "Early Neonatal",
            1: "Late Neonatal",
            2: "1-5 months",
            3: "6-11 months",
            4: "12 to 23 months",
            5: "2 to 4",
        },
        "age_group_id": {0: 2, 1: 3, 2: 388, 3: 389, 4: 238, 5: 34},
    }
    return pd.DataFrame(AGE_BINS_RAW_DICT)


# Age bins as processed by get_age_bins
AGE_BINS_EXPECTED_DICT = {
    "age_start": {0: 0.0, 1: 0.01917808, 2: 0.07671233, 3: 0.5, 4: 1.0, 5: 2.0},
    "age_end": {0: 0.01917808, 1: 0.07671233, 2: 0.5, 3: 1.0, 4: 2.0, 5: 5.0},
    "age_group_name": {
        0: "early_neonatal",
        1: "late_neonatal",
        2: "1-5_months",
        3: "6-11_months",
        4: "12_to_23_months",
        5: "2_to_4",
    },
    "age_group_id": {0: 2, 1: 3, 2: 388, 3: 389, 4: 238, 5: 34},
}

# Population table for mapper testing
FAKE_POP_AGE_DICT = {
    "age": {0: 0.01, 1: 0.45, 2: 1.01, 3: 1.99, 4: 2.02},
}

# Series of expected age_bin intervals for mapper testing
FAKE_POP_AGE_GROUP_EXPECTED_SERIES = pd.Series(
    {
        0: "early_neonatal",
        1: "1-5_months",
        2: "12_to_23_months",
        3: "12_to_23_months",
        4: "2_to_4",
    },
    name="age_group",
)

FAKE_POP_EVENT_TIME = {
    "year": {
        0: pd.to_datetime("1/1/2045"),
        1: pd.to_datetime("1/1/2045"),
        2: pd.to_datetime("1/1/2045"),
        3: pd.to_datetime("1/1/2045"),
        4: pd.to_datetime("1/1/2045"),
    },
}


def test_results_stratifier_register_stratifications(mocker):
    """Test that ResultsStratifier.register_stratifications registers expected stratifications
    and only the expected stratifications."""
    builder = mocker.Mock()
    builder.data.load = fake_data_load_population_age_bins
    builder.configuration.population.initialization_age_min = 0.0
    builder.configuration.population.untracking_age = 5.0
    builder.configuration.time.start.year = 2022
    builder.configuration.time.end.year = 2025
    years_list = ["2022", "2023", "2024", "2025"]
    age_group_names_list = [
        "early_neonatal",
        "late_neonatal",
        "1-5_months",
        "6-11_months",
        "12_to_23_months",
        "2_to_4",
    ]
    mocker.patch.object(builder, "results.register_stratification")
    builder.results.register_stratification = mocker.MagicMock()
    rs = ResultsStratifier()

    builder.results.register_stratification.assert_not_called()

    rs.setup(builder)  # setup calls register_stratifications()

    builder.results.register_stratification.assert_any_call(
        "age_group",
        age_group_names_list,
        mapper=rs.map_age_groups,
        is_vectorized=True,
        requires_columns=["age"],
    )
    builder.results.register_stratification.assert_any_call(
        "current_year",
        years_list,
        mapper=rs.map_year,
        is_vectorized=True,
        requires_columns=["current_time"],
    )
    # builder.results.register_stratification.assert_any_call(
    #     "event_year",
    #     years_list,
    #     mapper=rs.map_year,
    #     is_vectorized=True,
    #     requires_columns=["event_time"],
    # )
    # builder.results.register_stratification.assert_any_call(
    #     "entrance_year",
    #     years_list,
    #     mapper=rs.map_year,
    #     is_vectorized=True,
    #     requires_columns=["entrance_time"],
    # )
    # TODO [MIC-4803]: Known bug with this registration
    # builder.results.register_stratification.assert_any_call(
    #     "exit_year",
    #     years_list + ["nan"],
    #     mapper=rs.map_year,
    #     is_vectorized=True,
    #     requires_columns=["exit_time"],
    # )
    builder.results.register_stratification.assert_any_call(
        "sex", ["Female", "Male"], requires_columns=["sex"]
    )
    assert builder.results.register_stratification.call_count == 3


def test_results_stratifier_map_age_groups():
    """Test that ages of the population are mapped to intervals as expected."""
    pop = pd.DataFrame(FAKE_POP_AGE_DICT)
    rs = ResultsStratifier()
    rs.age_bins = pd.DataFrame(AGE_BINS_EXPECTED_DICT)
    mapped_pop = rs.map_age_groups(pop)
    pd.testing.assert_series_equal(
        mapped_pop,
        FAKE_POP_AGE_GROUP_EXPECTED_SERIES,
        check_dtype=False,
        check_categorical=False,
    )


def test_results_stratifier_map_year():
    """Test that datetimes are mapped to the correct year."""
    pop = pd.DataFrame(FAKE_POP_EVENT_TIME)
    rs = ResultsStratifier()
    the_year = rs.map_year(pop)
    assert (the_year == "2045").all()


def test_results_stratifier_get_age_bins(mocker):
    """Test that get_age_bins produces expected age_bins DataFrame."""
    builder = mocker.Mock()
    builder.data.load = fake_data_load_population_age_bins
    builder.configuration.population.initialization_age_min = 0.0
    builder.configuration.population.untracking_age = 5.0
    builder.configuration.time.start.year = 2022
    builder.configuration.time.end.year = 2025

    rs = ResultsStratifier()
    age_bins = rs.get_age_bins(builder)

    assert age_bins.equals(pd.DataFrame(AGE_BINS_EXPECTED_DICT))
