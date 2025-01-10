import numpy as np
import pandas as pd
import pytest

from tests.risks.test_effect import _setup_risk_effect_simulation
from tests.test_utilities import make_age_bins
from vivarium_public_health.risks.implementations.low_birth_weight_and_short_gestation import (
    LBWSGDistribution,
    LBWSGRisk,
    LBWSGRiskEffect,
)
from vivarium_public_health.utilities import to_snake_case


@pytest.mark.parametrize(
    "description, expected_age_values, expected_weight_values",
    [
        (
            "Neonatal preterm and LBWSG (estimation years) - [0, 24) wks, [0, 500) g",
            (0.0, 24.0),
            (0.0, 500.0),
        ),
        (
            "Neonatal preterm and LBWSG (estimation years) - [40, 42+] wks, [2000, 2500) g",
            (40.0, 42.0),
            (2000.0, 2500.0),
        ),
        (
            "Neonatal preterm and LBWSG (estimation years) - [34, 36) wks, [4000, 9999] g",
            (34.0, 36.0),
            (4000.0, 9999.0),
        ),
    ],
)
def test_parsing_lbwsg_descriptions(description, expected_weight_values, expected_age_values):
    weight_interval = LBWSGDistribution._parse_description("birth_weight", description)
    age_interval = LBWSGDistribution._parse_description("gestational_age", description)
    assert weight_interval.left == expected_weight_values[0]
    assert weight_interval.right == expected_weight_values[1]
    assert age_interval.left == expected_age_values[0]
    assert age_interval.right == expected_age_values[1]


def test_lbwsg_risk_effect_rr_pipeline(
    base_config, base_plugins, mocker, mock_rr_interpolators
):

    risk = LBWSGRisk()
    lbwsg_effect = LBWSGRiskEffect("cause.test_cause.cause_specific_mortality_rate")

    # Add mock data to artifact
    categories = {
        "cat81": "Neonatal preterm and LBWSG (estimation years) - [28, 30) wks, [2500, 3000) g",
        "cat82": "Neonatal preterm and LBWSG (estimation years) - [28, 30) wks, [3000, 3500) g",
    }
    # Create exposure with matching demograph index as age_bins
    age_bins = make_age_bins()
    agees = age_bins.drop(columns="age_group_name")
    exposure_data = make_categorical_exposure_data(agees)

    # Add data dict to add to artifact
    data = {
        f"{risk.name}.exposure": exposure_data,
        f"{risk.name}.population_attributable_fraction": 0,
        f"{risk.name}.categories": categories,
        f"{risk.name}.relative_risk_interpolator": mock_rr_interpolators,
    }

    # Only have neontal age groups
    age_start = 0.0
    age_end = 28 / 365.0
    base_config.update(
        {
            "population": {
                "initialization_age_start": age_start,
                "initialization_age_max": age_end,
            }
        }
    )
    sim = _setup_risk_effect_simulation(base_config, base_plugins, risk, lbwsg_effect, data)
    pop = sim.get_population()

    expected_pipeline_name = (
        f"effect_of_{lbwsg_effect.risk.name}_on_{lbwsg_effect.target.name}.relative_risk"
    )
    assert expected_pipeline_name in sim.list_values()

    # Get age group names to lookup rr interpolator later
    def map_age_groups(value):
        for i, row in age_bins.iterrows():
            if row["age_start"] <= value <= row["age_end"]:
                return row["age_group_name"]
        return None

    mapped_age_groups = pop["age"].apply(map_age_groups)
    mapped_age_groups = mapped_age_groups.apply(to_snake_case)
    sim_data = pop[["sex", "birth_weight_exposure", "gestational_age_exposure"]].copy()
    sim_data["age_group_name"] = mapped_age_groups

    # Test the 4 different demographic groups
    for sex in ["Male", "Female"]:
        for age_group_name in ["early_neonatal", "late_neonatal"]:
            interpolator = lbwsg_effect.interpolator[sex, age_group_name]
            demo_idx = sim_data.index[
                (sim_data["sex"] == sex) & (sim_data["age_group_name"] == age_group_name)
            ]
            sub_pop = sim_data.loc[demo_idx]
            actual_rr = sim.get_value(expected_pipeline_name)(demo_idx)
            sub_pop["expected_rr"] = np.exp(
                interpolator(
                    sub_pop["gestational_age_exposure"],
                    sub_pop["birth_weight_exposure"],
                    grid=False,
                )
            )
            assert (actual_rr == sub_pop["expected_rr"]).all()


def make_categorical_exposure_data(data: pd.DataFrame) -> pd.DataFrame:
    # Takes age gropus and adds sex, years, categories, and values
    exposure_dfs = []
    for year in range(1990, 2017):
        tmp = data.copy()
        tmp["year_start"] = year
        tmp["year_end"] = year + 1
        p_81 = tmp.copy()
        p_81["parameter"] = "cat81"
        p_81["value"] = 0.75
        p_82 = tmp.copy()
        p_82["parameter"] = "cat82"
        p_82["value"] = 0.25
        categories_df = pd.concat([p_81, p_82])
        male_tmp = categories_df.copy()
        male_tmp["sex"] = "Male"
        female_tmp = categories_df.copy()
        female_tmp["sex"] = "Female"
        age_sex_df = pd.concat([male_tmp, female_tmp])
        exposure_dfs.append(age_sex_df)

    return pd.concat(exposure_dfs)
