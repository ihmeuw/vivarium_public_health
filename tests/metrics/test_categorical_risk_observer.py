import numpy as np
import pandas as pd
import pytest
from vivarium import InteractiveContext
from vivarium.testing_utilities import TestPopulation, build_table

from vivarium_public_health.metrics.risk import CategoricalRiskObserver
from vivarium_public_health.metrics.stratification import (
    ResultsStratifier as ResultsStratifier_,
)
from vivarium_public_health.risks.base_risk import Risk
from vivarium_public_health.utilities import to_years


@pytest.fixture
def categorical_risk():
    year_start = 1990
    year_end = 2010
    risk = "test_risk"
    risk_data = dict()
    exposure_data = build_table(
        0.25, year_start, year_end, ["age", "year", "sex", "cat1", "cat2", "cat3", "cat4"]
    ).melt(
        id_vars=("age_start", "age_end", "year_start", "year_end", "sex"),
        var_name="parameter",
        value_name="value",
    )

    risk_data["exposure"] = exposure_data
    risk_data["categories"] = {
        "cat1": "severe",
        "cat2": "moderate",
        "cat3": "mild",
        "cat4": "unexposed",
    }
    risk_data["distribution"] = "ordered_polytomous"
    return Risk(f"risk_factor.{risk}"), risk_data


# Subclass of ResultsStratifier for integration testing
class ResultsStratifier(ResultsStratifier_):
    configuration_defaults = {
        "stratification": {
            "default": ["age_group", "sex"],
        }
    }


def test_observation_registration(base_config, base_plugins, categorical_risk):
    """Test that all expected observation keys appear as expected in the results."""
    risk, risk_data = categorical_risk
    observer = CategoricalRiskObserver(f"{risk.risk.name}")
    simulation = InteractiveContext(
        components=[
            TestPopulation(),
            ResultsStratifier(),
            risk,
            observer,
        ],
        configuration=base_config,
        plugin_configuration=base_plugins,
        setup=False,
    )
    simulation.configuration.update(
        {
            "stratification": {
                "test_risk": {
                    "exclude": ["age_group"],
                }
            }
        }
    )

    for key, value in risk_data.items():
        simulation._data.write(f"risk_factor.test_risk.{key}", value)

    simulation.setup()
    pop = simulation.get_population()
    simulation.step()
    results = simulation.get_value("metrics")
    expected_observations = [
        "MEASURE_test_risk_cat1_person_time_SEX_Female",
        "MEASURE_test_risk_cat1_person_time_SEX_Male",
        "MEASURE_test_risk_cat2_person_time_SEX_Female",
        "MEASURE_test_risk_cat2_person_time_SEX_Male",
        "MEASURE_test_risk_cat3_person_time_SEX_Female",
        "MEASURE_test_risk_cat3_person_time_SEX_Male",
        "MEASURE_test_risk_cat4_person_time_SEX_Female",
        "MEASURE_test_risk_cat4_person_time_SEX_Male",
    ]
    for v in expected_observations:
        assert v in results(pop.index).keys()


def test_observation_correctness(base_config, base_plugins, categorical_risk):
    """Test that person time appear as expected in the results."""
    time_step = pd.Timedelta(days=base_config.time.step_size)
    risk, risk_data = categorical_risk
    observer = CategoricalRiskObserver(f"{risk.risk.name}")
    simulation = InteractiveContext(
        components=[
            TestPopulation(),
            ResultsStratifier(),
            risk,
            observer,
        ],
        configuration=base_config,
        plugin_configuration=base_plugins,
        setup=False,
    )
    simulation.configuration.update(
        {
            "stratification": {
                "test_risk": {
                    "exclude": ["age_group"],
                }
            }
        }
    )

    for key, value in risk_data.items():
        simulation._data.write(f"risk_factor.test_risk.{key}", value)

    simulation.setup()
    pop = simulation.get_population()
    results = simulation.get_value("metrics")

    simulation.step()

    exposure = simulation._values.get_value("test_risk.exposure")(pop.index)
    exposure_categories = risk_data["categories"].keys()

    for category in exposure_categories:
        for sex in ["Male", "Female"]:
            observation = f"MEASURE_test_risk_{category}_person_time_SEX_{sex}"
            expected_person_time = sum(
                (exposure == category) & (pop["sex"] == sex)
            ) * to_years(time_step)
            assert np.isclose(
                results(pop.index)[observation], expected_person_time, rtol=0.001
            )
