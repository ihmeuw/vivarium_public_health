import numpy as np
import pandas as pd
import pytest
from layered_config_tree import ConfigurationError
from vivarium import InteractiveContext
from vivarium.testing_utilities import TestPopulation

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
    age_interval, weight_interval = LBWSGDistribution._parse_description(description)
    assert weight_interval.left == expected_weight_values[0]
    assert weight_interval.right == expected_weight_values[1]
    assert age_interval.left == expected_age_values[0]
    assert age_interval.right == expected_age_values[1]


def test_lbwsg_risk_effect_rr_pipeline(base_config, base_plugins, mock_rr_interpolators):

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
    # Have to match age bins and rr data to make age intervals
    rr_data = make_categorical_data(agees)
    # Exposure data used for risk component
    birth_exposure = make_categorical_data(agees)
    exposure = birth_exposure.copy()
    exposure.loc[exposure["value"] == 0.75, "value"] = 0.65
    exposure.loc[exposure["value"] == 0.25, "value"] = 0.35

    # Add data dict to add to artifact
    data = {
        f"{risk.name}.birth_exposure": birth_exposure,
        f"{risk.name}.exposure": exposure,
        f"{risk.name}.relative_risk": rr_data,
        f"{risk.name}.population_attributable_fraction": 0,
        f"{risk.name}.categories": categories,
        f"{risk.name}.relative_risk_interpolator": mock_rr_interpolators,
    }

    # Only have neontal age groups
    age_start = 0.0
    age_end = 1.0
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
    # Verify exposure is used instead of birth_exposure since age end is 1.0
    # Check values of pipeline match birth exposure data since age_end is 0.0
    exposure_pipeline_values = sim.get_value(
        "risk_factor.low_birth_weight_and_short_gestation.exposure_parameters"
    )(pop.index)
    assert isinstance(exposure_pipeline_values, pd.DataFrame)
    assert "cat81" in exposure_pipeline_values.columns
    assert "cat82" in exposure_pipeline_values.columns
    assert (exposure_pipeline_values["cat81"] == 0.65).all()
    assert (exposure_pipeline_values["cat82"] == 0.35).all()

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
        for age_group_name in ["early_neonatal", "late_neonatal", "post_neonatal"]:
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
            if age_group_name == "post_neonatal":
                assert (actual_rr == 1.0).all()


@pytest.mark.parametrize("age_end", [0.0, 1.0])
def test_use_exposure(base_config, base_plugins, mock_rr_interpolators, age_end):
    risk = LBWSGRisk()
    lbwsg_effect = LBWSGRiskEffect("cause.test_cause.cause_specific_mortality_rate")

    # Add mock data to artifact
    categories = {
        "cat81": "Neonatal preterm and LBWSG (estimation years) - [28, 30) wks, [2500, 3000) g",
        "cat82": "Neonatal preterm and LBWSG (estimation years) - [28, 30) wks, [3000, 3500) g",
    }
    # Create exposure with matching demograph index as age_bins
    age_bins = make_age_bins()
    ages = age_bins.drop(columns="age_group_name")
    # Have to match age bins and rr data to make age intervals
    rr_data = make_categorical_data(ages)
    # Format birth exposure data
    birth_exposure = pd.DataFrame(
        {
            "sex": ["Male", "Female", "Male", "Female"],
            "year_start": [2021, 2021, 2021, 2021],
            "year_end": [2022, 2022, 2022, 2022],
            "parameter": ["cat81", "cat81", "cat82", "cat82"],
            "value": [0.75, 0.75, 0.25, 0.25],
        }
    )
    exposure = birth_exposure.copy()
    exposure["value"] = [0.65, 0.65, 0.35, 0.35]

    # Add data dict to add to artifact
    data = {
        f"{risk.name}.birth_exposure": birth_exposure,
        f"{risk.name}.exposure": exposure,
        f"{risk.name}.relative_risk": rr_data,
        f"{risk.name}.population_attributable_fraction": 0,
        f"{risk.name}.categories": categories,
        f"{risk.name}.relative_risk_interpolator": mock_rr_interpolators,
    }

    # Only have neontal age groups
    base_config.update(
        {
            "population": {
                "initialization_age_start": 0.0,
                "initialization_age_max": age_end,
            },
        }
    )
    sim = _setup_risk_effect_simulation(base_config, base_plugins, risk, lbwsg_effect, data)
    pop = sim.get_population()
    # Check values of pipeline match birth exposure data since age_end is 0.0
    exposure_pipeline_values = sim.get_value(
        "risk_factor.low_birth_weight_and_short_gestation.exposure_parameters"
    )(pop.index)
    assert isinstance(exposure_pipeline_values, pd.DataFrame)
    assert "cat81" in exposure_pipeline_values.columns
    assert "cat82" in exposure_pipeline_values.columns
    exposure_values = {
        0.0: {"cat81": 0.75, "cat82": 0.25},
        1.0: {"cat81": 0.65, "cat82": 0.35},
    }
    assert (exposure_pipeline_values["cat81"] == exposure_values[age_end]["cat81"]).all()
    assert (exposure_pipeline_values["cat82"] == exposure_values[age_end]["cat82"]).all()

    # Assert LBWSG birth exposure columns were created
    assert "birth_weight_exposure" in pop.columns
    assert "gestational_age_exposure" in pop.columns


@pytest.mark.parametrize("exposure_key", ["birth_exposure", "exposure", "missing"])
def test_lbwsg_exposure_data_logging(exposure_key, base_config, mocker, caplog) -> None:
    risk = LBWSGRisk()

    # Add mock data to artifact
    # Format birth exposure data
    exposure_data = pd.DataFrame(
        {
            "sex": ["Male", "Female", "Male", "Female"],
            "year_start": [2021, 2021, 2021, 2021],
            "year_end": [2022, 2022, 2022, 2022],
            "parameter": ["cat81", "cat81", "cat82", "cat82"],
            "value": [0.75, 0.75, 0.25, 0.25],
        }
    )

    # Only have neontal age groups
    if exposure_key == "birth_exposure":
        age_end = 0.0
    else:
        age_end = 1.0

    if exposure_key != "missing":
        override_config = {
            "population": {
                "initialization_age_start": 0.0,
                "initialization_age_max": age_end,
            },
            risk.name: {
                "data_sources": {
                    exposure_key: exposure_data,
                }
            },
        }
    else:
        override_config = {
            "population": {
                "initialization_age_start": 0.0,
                "initialization_age_max": age_end,
            },
        }

    # Patch get_category intervals so we do not need the mock artifact
    mocker.patch(
        "vivarium_public_health.risks.implementations.low_birth_weight_and_short_gestation.LBWSGDistribution.get_category_intervals"
    )
    assert not caplog.records
    if exposure_key != "missing":
        missing_key = "exposure" if exposure_key == "birth_exposure" else "birth_exposure"
        _ = InteractiveContext(
            base_config,
            components=[TestPopulation(), risk],
            configuration=override_config,
        )
        assert f"The data for LBWSG {missing_key} is missing from the simulation"
    else:
        with pytest.raises(
            ConfigurationError,
            match="The LBWSG distribution requires either 'birth_exposure' or 'exposure' data to be "
            "available in the simulation.",
        ):
            InteractiveContext(
                base_config,
                components=[TestPopulation(), risk],
                configuration=override_config,
            )


def make_categorical_data(data: pd.DataFrame) -> pd.DataFrame:
    # Takes age gropus and adds sex, years, categories, and values
    dfs = []
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
        dfs.append(age_sex_df)

    return pd.concat(dfs)
