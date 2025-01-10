import pandas as pd
import pytest

from tests.test_utilities import build_table_with_age
from vivarium_public_health.risks.base_risk import Risk


def make_test_data_table(values: list, parameter="cat") -> pd.DataFrame:
    year_start = 1990  # same as the base config
    year_end = 2010

    if len(values) == 1:
        df = build_table_with_age(
            values[0], parameter_columns={"year": (year_start, year_end)}
        )
    else:
        cats = (
            [f"{parameter}{i+1}" for i in range(len(values))]
            if parameter == "cat"
            else parameter
        )
        df = []
        for cat, value in zip(cats, values):
            df.append(
                build_table_with_age(
                    [cat, value],
                    parameter_columns={"year": (year_start, year_end)},
                    value_columns=["parameter", "value"],
                )
            )
        df = pd.concat(df)
    return df


@pytest.fixture(scope="module")
def continuous_risk():
    year_start = 1990
    year_end = 2010
    risk = "test_risk"
    risk_data = dict()
    exposure_mean = make_test_data_table([130])
    exposure_sd = make_test_data_table([15])
    affected_causes = ["test_cause_1", "test_cause_2"]
    rr_data = []
    paf_data = []
    for cause in affected_causes:
        rr_data.append(
            build_table_with_age(
                [1.01, cause],
                parameter_columns={"year": (year_start, year_end)},
                value_columns=["value", "cause"],
            ).melt(
                id_vars=("age_start", "age_end", "year_start", "year_end", "sex", "cause"),
                var_name="parameter",
                value_name="value",
            )
        )
        paf_data.append(
            build_table_with_age(
                [1, cause],
                parameter_columns={"year": (year_start, year_end)},
                value_columns=["value", "cause"],
            )
        )
    rr_data = pd.concat(rr_data)
    paf_data = pd.concat(paf_data)
    paf_data["affected_measure"] = "incidence_rate"
    rr_data["affected_measure"] = "incidence_rate"
    risk_data["exposure"] = exposure_mean
    risk_data["exposure_standard_deviation"] = exposure_sd
    risk_data["relative_risk"] = rr_data
    risk_data["population_attributable_fraction"] = paf_data
    risk_data["affected_causes"] = affected_causes
    risk_data["affected_risk_factors"] = []

    tmred = {
        "distribution": "uniform",
        "min": 110.0,
        "max": 115.0,
        "inverted": False,
    }
    exposure_parameters = {
        "scale": 10.0,
        "max_rr": 200.0,
        "max_val": 300.0,
        "min_val": 50.0,
    }
    tmrel = 0.5 * (tmred["max"] + tmred["min"])
    risk_data["tmred"] = tmred
    risk_data["tmrel"] = tmrel
    risk_data["exposure_parameters"] = exposure_parameters
    risk_data["distribution"] = "normal"

    return Risk(f"risk_factor.{risk}"), risk_data


@pytest.fixture(scope="module")
def dichotomous_risk():
    year_start = 1990
    year_end = 2010
    risk = "test_risk"
    risk_data = dict()
    exposure_data = build_table_with_age(
        0.5,
        parameter_columns={"year": (year_start, year_end)},
        value_columns=["cat1", "cat2"],
    ).melt(
        id_vars=("age_start", "age_end", "year_start", "year_end", "sex"),
        var_name="parameter",
        value_name="value",
    )

    affected_causes = ["test_cause_1", "test_cause_2"]
    rr_data = []
    paf_data = []
    for cause in affected_causes:
        rr_data.append(
            build_table_with_age(
                [1.01, 1, cause],
                parameter_columns={"year": (year_start, year_end)},
                value_columns=["cat1", "cat2", "cause"],
            ).melt(
                id_vars=("age_start", "age_end", "year_start", "year_end", "sex", "cause"),
                var_name="parameter",
                value_name="value",
            )
        )
        paf_data.append(
            build_table_with_age(
                [1, cause],
                parameter_columns={"year": (year_start, year_end)},
                value_columns=["value", "cause"],
            )
        )
    rr_data = pd.concat(rr_data)
    paf_data = pd.concat(paf_data)
    paf_data["affected_measure"] = "incidence_rate"
    rr_data["affected_measure"] = "incidence_rate"
    risk_data["exposure"] = exposure_data
    risk_data["relative_risk"] = rr_data
    risk_data["population_attributable_fraction"] = paf_data
    risk_data["affected_causes"] = affected_causes
    risk_data["affected_risk_factors"] = []
    incidence_rate = build_table_with_age(
        0.01, parameter_columns={"year": (year_start, year_end)}
    )
    risk_data["incidence_rate"] = incidence_rate
    risk_data["distribution"] = "dichotomous"
    return Risk(f"risk_factor.{risk}"), risk_data


@pytest.fixture(scope="module")
def polytomous_risk():
    year_start = 1990
    year_end = 2010
    risk = "test_risk"
    risk_data = dict()
    exposure_data = build_table_with_age(
        0.25,
        parameter_columns={"year": (year_start, year_end)},
        value_columns=["cat1", "cat2", "cat3", "cat4"],
    ).melt(
        id_vars=("age_start", "age_end", "year_start", "year_end", "sex"),
        var_name="parameter",
        value_name="value",
    )

    affected_causes = ["test_cause_1", "test_cause_2"]
    rr_data = []
    paf_data = []
    for cause in affected_causes:
        rr_data.append(
            build_table_with_age(
                [1.03, 1.02, 1.01, 1, cause],
                parameter_columns={"year": (year_start, year_end)},
                value_columns=["cat1", "cat2", "cat3", "cat4", "cause"],
            ).melt(
                id_vars=("age_start", "age_end", "year_start", "year_end", "sex", "cause"),
                var_name="parameter",
                value_name="value",
            )
        )
        paf_data.append(
            build_table_with_age(
                [1, cause],
                parameter_columns={"year": (year_start, year_end)},
                value_columns=["value", "cause"],
            )
        )
    rr_data = pd.concat(rr_data)
    paf_data = pd.concat(paf_data)
    paf_data["affected_measure"] = "incidence_rate"
    rr_data["affected_measure"] = "incidence_rate"
    risk_data["exposure"] = exposure_data
    risk_data["relative_risk"] = rr_data
    risk_data["population_attributable_fraction"] = paf_data
    risk_data["affected_causes"] = affected_causes
    risk_data["affected_risk_factors"] = []
    incidence_rate = build_table_with_age(
        0.01, parameter_columns={"year": (year_start - 1, year_end)}
    )
    risk_data["incidence_rate"] = incidence_rate
    risk_data["distribution"] = "polytomous"
    return Risk(f"risk_factor.{risk}"), risk_data


@pytest.fixture(scope="module")
def coverage_gap():
    year_start = 1990
    year_end = 2010
    cg = "test_coverage_gap"
    cg_data = dict()
    cg_exposed = 0.6
    cg_exposure_data = build_table_with_age(
        [cg_exposed, 1 - cg_exposed],
        parameter_columns={"year": (year_start, year_end)},
        value_columns=["cat1", "cat2"],
    ).melt(
        id_vars=(
            "age_start",
            "age_end",
            "year_start",
            "year_end",
            "sex",
        ),
        var_name="parameter",
        value_name="value",
    )

    rr = 2
    rr_data = build_table_with_age(
        [rr, 1],
        parameter_columns={"year": (year_start - 1, year_end)},
        value_columns=["cat1", "cat2"],
    ).melt(
        id_vars=("age_start", "age_end", "year_start", "year_end", "sex"),
        var_name="parameter",
        value_name="value",
    )

    # paf is (sum(exposure(category)*rr(category) -1 )/ (sum(exposure(category)* rr(category)
    paf = (rr * cg_exposed + (1 - cg_exposed) - 1) / (rr * cg_exposed + (1 - cg_exposed))

    paf_data = build_table_with_age(
        paf,
        parameter_columns={"year": (year_start - 1, year_end)},
        value_columns=["population_attributable_fraction"],
    ).melt(
        id_vars=("age_start", "age_end", "year_start", "year_end", "sex"),
        var_name="population_attributable_fraction",
        value_name="value",
    )

    paf_data["risk_factor"] = "test_risk"
    paf_data["affected_measure"] = "exposure_parameters"
    rr_data["affected_measure"] = "exposure_parameters"
    cg_data["exposure"] = cg_exposure_data
    rr_data["risk_factor"] = "test_risk"
    cg_data["relative_risk"] = rr_data
    cg_data["population_attributable_fraction"] = paf_data
    cg_data["affected_causes"] = []
    cg_data["affected_risk_factors"] = ["test_risk"]
    cg_data["distribution"] = "dichotomous"
    return Risk(f"coverage_gap.{cg}"), cg_data


@pytest.fixture
def mock_rr_interpolators() -> pd.DataFrame:
    rr_interpolators = pd.read_csv("tests/data/rr_interpolator.csv")
    idx_cols = [col for col in rr_interpolators.columns if "draw" not in col]
    rr_interpolators = rr_interpolators.rename(columns={"draw_0": "value"})
    return rr_interpolators
