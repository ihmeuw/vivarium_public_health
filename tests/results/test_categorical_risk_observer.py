import itertools

import numpy as np
import pandas as pd
import pytest
from vivarium import InteractiveContext
from vivarium.testing_utilities import TestPopulation

from tests.test_utilities import build_table_with_age
from vivarium_public_health.results.columns import COLUMNS
from vivarium_public_health.results.risk import CategoricalRiskObserver
from vivarium_public_health.results.stratification import ResultsStratifier
from vivarium_public_health.risks.base_risk import Risk
from vivarium_public_health.utilities import to_years


@pytest.fixture
def categorical_risk():
    year_start = 1990
    year_end = 2010
    risk = "test_risk"
    risk_data = dict()
    exposure_data = build_table_with_age(
        0.25,
        parameter_columns={
            "year": (year_start, year_end),
        },
        value_columns=["cat1", "cat2", "cat3", "cat4"],
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


@pytest.fixture()
def simulation_after_one_step(base_config, base_plugins, categorical_risk):
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
                    "include": ["sex"],
                }
            }
        }
    )

    for key, value in risk_data.items():
        simulation._data.write(f"risk_factor.test_risk.{key}", value)

    simulation.setup()
    simulation.step()
    simulation.finalize()
    simulation.report()

    return simulation


def test_observation_registration(simulation_after_one_step):
    """Test that all expected observation stratifications appear in the results."""
    results = simulation_after_one_step.get_results()
    assert set(results) == set(["person_time_test_risk"])

    person_time = results["person_time_test_risk"]

    assert set(zip(person_time[COLUMNS.SUB_ENTITY], person_time["sex"])) == set(
        itertools.product(*[["cat1", "cat2", "cat3", "cat4"], ["Female", "Male"]])
    )


def test_observation_correctness(base_config, simulation_after_one_step, categorical_risk):
    """Test that person time appear as expected in the results."""
    time_step = pd.Timedelta(days=base_config.time.step_size)

    _, risk_data = categorical_risk
    exposure_categories = risk_data["categories"].keys()

    pop = simulation_after_one_step.get_population()
    exposure = simulation_after_one_step.get_value("test_risk.exposure")(pop.index)

    results = simulation_after_one_step.get_results()
    assert set(results) == set(["person_time_test_risk"])
    results = results["person_time_test_risk"]

    # Check columns
    assert set(results.columns) == set(
        [
            "sex",
            COLUMNS.MEASURE,
            COLUMNS.ENTITY_TYPE,
            COLUMNS.ENTITY,
            COLUMNS.SUB_ENTITY,
            COLUMNS.VALUE,
        ]
    )

    assert (results[COLUMNS.MEASURE] == "person_time").all()
    assert (results[COLUMNS.ENTITY_TYPE] == "rei").all()
    assert (results[COLUMNS.ENTITY] == "test_risk").all()
    for category in exposure_categories:
        for sex in ["Male", "Female"]:
            expected_person_time = sum(
                (exposure == category) & (pop["sex"] == sex)
            ) * to_years(time_step)
            actual_person_time = results.loc[
                (results[COLUMNS.SUB_ENTITY] == category) & (results["sex"] == sex),
                COLUMNS.VALUE,
            ].values[0]
            assert np.isclose(expected_person_time, actual_person_time, rtol=0.001)


def test_different_results_per_risk(base_config, base_plugins, categorical_risk):
    """Test that each observer saves its own results."""

    risk, risk_data = categorical_risk
    risk_observer = CategoricalRiskObserver(f"{risk.risk.name}")

    # Set up a second risk factor
    another_risk = Risk("risk_factor.another_test_risk")
    another_risk_observer = CategoricalRiskObserver(f"{another_risk.risk.name}")

    simulation = InteractiveContext(
        components=[
            TestPopulation(),
            ResultsStratifier(),
            risk,
            risk_observer,
            another_risk,
            another_risk_observer,
        ],
        configuration=base_config,
        plugin_configuration=base_plugins,
        setup=False,
    )
    for key, value in risk_data.items():
        simulation._data.write(f"risk_factor.test_risk.{key}", value)
        simulation._data.write(f"risk_factor.another_test_risk.{key}", value)

    assert not simulation.get_results()
    simulation.setup()
    simulation.step()
    results = simulation.get_results()
    assert set(results) == set(["person_time_test_risk", "person_time_another_test_risk"])
    assert (
        results["person_time_test_risk"]["value"]
        != results["person_time_another_test_risk"]["value"]
    ).all()


@pytest.mark.parametrize("exclusions", [[], ["cat1"], ["cat1", "cat4"]])
def test_category_exclusions(base_config, base_plugins, categorical_risk, exclusions):
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
                    "include": ["sex"],
                },
                "excluded_categories": {
                    "test_risk": exclusions,
                },
            },
        },
    )

    for key, value in risk_data.items():
        simulation._data.write(f"risk_factor.test_risk.{key}", value)

    simulation.setup()
    simulation.step()
    results = simulation.get_results()["person_time_test_risk"]
    assert set(results["sub_entity"]) == {"cat1", "cat2", "cat3", "cat4"} - set(exclusions)
