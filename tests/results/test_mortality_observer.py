import itertools
from collections import Counter

import numpy as np
import pytest
from vivarium import InteractiveContext
from vivarium.testing_utilities import TestPopulation

from tests.test_utilities import build_table_with_age
from vivarium_public_health.disease import DiseaseModel, DiseaseState
from vivarium_public_health.disease.state import SusceptibleState
from vivarium_public_health.population import Mortality
from vivarium_public_health.results import MortalityObserver
from vivarium_public_health.results.columns import COLUMNS
from vivarium_public_health.results.stratification import ResultsStratifier


def disease_with_excess_mortality(base_config, disease_name, emr_value) -> DiseaseModel:
    year_start = base_config.time.start.year
    year_end = base_config.time.end.year
    healthy = SusceptibleState(disease_name, allow_self_transition=True)
    disease_get_data_funcs = {
        "disability_weight": lambda *_: build_table_with_age(
            0.0, parameter_columns={"year": (year_start - 1, year_end)}
        ),
        "prevalence": lambda *_: build_table_with_age(
            0.5, parameter_columns={"year": (year_start - 1, year_end)}
        ),
        "excess_mortality_rate": lambda *_: build_table_with_age(
            emr_value, parameter_columns={"year": (year_start - 1, year_end)}
        ),
    }
    with_condition = DiseaseState(disease_name, get_data_functions=disease_get_data_funcs)
    healthy.add_rate_transition(
        with_condition,
        get_data_functions={
            "incidence_rate": lambda *_: build_table_with_age(
                0.1, parameter_columns={"year": (year_start - 1, year_end)}
            )
        },
    )
    return DiseaseModel(disease_name, states=[healthy, with_condition])


@pytest.fixture()
def simulation_after_one_step(base_config, base_plugins):
    flu = disease_with_excess_mortality(base_config, "flu", 10)
    mumps = disease_with_excess_mortality(base_config, "mumps", 20)
    # TODO: Add test against using a RiskAttributableDisease in addition to a DiseaseModel

    simulation = InteractiveContext(
        components=[
            TestPopulation(),
            Mortality(),
            flu,
            mumps,
            ResultsStratifier(),
            MortalityObserver(),
        ],
        configuration=base_config,
        plugin_configuration=base_plugins,
        setup=False,
    )
    simulation.configuration.update(
        {
            "stratification": {
                "mortality": {
                    "include": ["sex"],
                }
            }
        }
    )

    year_start = base_config.time.start.year
    year_end = base_config.time.end.year
    acmr_data = build_table_with_age(
        0.5, parameter_columns={"year": (year_start - 1, year_end)}
    )
    simulation._data.write("cause.all_causes.cause_specific_mortality_rate", acmr_data)

    simulation.setup()
    simulation.step()

    return simulation


def _get_expected_results(simulation, expected_values=Counter()):
    """Get expected results given a simulation. If expected deaths, for example,
    are not provided, return the counts of deaths in this time step. If expected
    deaths are provided, return the counts of deaths in the Counter plus the counts
    for this time step.
    """
    pop = simulation.get_population()

    for cause in ["other_causes", "flu", "mumps"]:
        for sex in ["Male", "Female"]:
            deaths_observation = f"MEASURE_deaths_due_to_{cause}_SEX_{sex}"
            ylls_observation = f"MEASURE_ylls_due_to_{cause}_SEX_{sex}"

            died_of_cause = pop["cause_of_death"] == cause
            is_sex_of_interest = pop["sex"] == sex
            died_this_step = pop["exit_time"] == simulation._clock.time
            is_pop_of_interest = died_of_cause & is_sex_of_interest & died_this_step
            expected_values.update({deaths_observation: sum(is_pop_of_interest)})
            expected_values.update(
                {ylls_observation: sum(pop.loc[is_pop_of_interest, "years_of_life_lost"])}
            )

    return expected_values


def test_observation_registration(simulation_after_one_step):
    """Test that all expected observation stratifications appear in the results."""
    results = simulation_after_one_step.get_results()
    deaths = results["deaths"]
    ylls = results["ylls"]

    expected_stratifications = set(
        itertools.product(*[["other_causes", "flu", "mumps"], ["Female", "Male"]])
    )
    assert set(zip(deaths[COLUMNS.ENTITY], deaths["sex"])) == expected_stratifications
    assert set(zip(ylls[COLUMNS.ENTITY], ylls["sex"])) == expected_stratifications


def test_observation_correctness(simulation_after_one_step):
    """Test that deaths and YLLs appear as expected in the results."""
    results = simulation_after_one_step.get_results()
    expected = _get_expected_results(simulation_after_one_step)
    _assert_correctness(results, expected)

    # same test on second time step
    simulation_after_one_step.step()
    results = simulation_after_one_step.get_results()
    expected = _get_expected_results(simulation_after_one_step, expected)
    _assert_correctness(results, expected)

    # Check for all expected columns and stratifications
    for measure in ["deaths", "ylls"]:
        df = results[measure]

        # Check columns
        assert set(df.columns) == set(
            [
                "sex",
                COLUMNS.MEASURE,
                COLUMNS.ENTITY_TYPE,
                COLUMNS.ENTITY,
                COLUMNS.SUB_ENTITY,
                COLUMNS.VALUE,
            ]
        )
        assert (df[COLUMNS.MEASURE] == measure).all()
        assert (df[COLUMNS.ENTITY_TYPE] == "cause").all()
        assert set(df[COLUMNS.ENTITY]) == set(["other_causes", "flu", "mumps"])


def test_aggregation_configuration(base_config, base_plugins):

    observer = MortalityObserver()
    flu = disease_with_excess_mortality(base_config, "flu", 10)
    mumps = disease_with_excess_mortality(base_config, "mumps", 20)

    aggregate_sim = InteractiveContext(
        components=[
            TestPopulation(),
            Mortality(),
            flu,
            mumps,
            ResultsStratifier(),
            observer,
        ],
        configuration=base_config,
        plugin_configuration=base_plugins,
        setup=False,
    )
    aggregate_sim.configuration.update(
        {
            "stratification": {
                "mortality": {
                    "include": ["sex"],
                    "aggregate": True,
                }
            }
        }
    )

    year_start = base_config.time.start.year
    year_end = base_config.time.end.year
    acmr_data = build_table_with_age(
        0.5, parameter_columns={"year": (year_start - 1, year_end)}
    )
    aggregate_sim._data.write("cause.all_causes.cause_specific_mortality_rate", acmr_data)
    aggregate_sim.setup()
    aggregate_sim.step()

    results = aggregate_sim.get_results()
    deaths = results["deaths"]
    ylls = results["ylls"]

    expected_stratifications = set(itertools.product(*[["all_causes"], ["Female", "Male"]]))

    assert set(zip(deaths[COLUMNS.ENTITY], deaths["sex"])) == expected_stratifications
    assert set(zip(ylls[COLUMNS.ENTITY], ylls["sex"])) == expected_stratifications


@pytest.mark.parametrize(
    "exclusions",
    [[], ["other_causes"], ["other_causes", "flu"]],
)
def test_category_exclusions(base_config, base_plugins, exclusions):
    flu = disease_with_excess_mortality(base_config, "flu", 10)
    mumps = disease_with_excess_mortality(base_config, "mumps", 20)

    simulation = InteractiveContext(
        components=[
            TestPopulation(),
            Mortality(),
            flu,
            mumps,
            ResultsStratifier(),
            MortalityObserver(),
        ],
        configuration=base_config,
        plugin_configuration=base_plugins,
        setup=False,
    )
    simulation.configuration.update(
        {
            "stratification": {
                "mortality": {
                    "include": ["sex"],
                },
                "excluded_categories": {
                    "cause_of_death": exclusions,
                },
            },
        },
    )

    year_start = base_config.time.start.year
    year_end = base_config.time.end.year
    acmr_data = build_table_with_age(
        0.5, parameter_columns={"year": (year_start - 1, year_end)}
    )
    simulation._data.write("cause.all_causes.cause_specific_mortality_rate", acmr_data)

    simulation.setup()
    simulation.step()
    results = simulation.get_results()["deaths"]
    assert set(results["sub_entity"]) == {"other_causes", "flu", "mumps"} - set(exclusions)


##################
# Helper functions
##################


def _assert_correctness(results, expected):

    for observation in expected:
        measure = observation.split("_due_to_")[0].strip("MEASURE_")
        cause = observation.split("_due_to_")[1].split("_SEX_")[0]
        sex = observation.split("SEX_")[1]
        df = results[measure]
        assert (
            df.loc[(df["sex"] == sex) & (df["entity"] == cause), "value"]
            == expected[observation]
        ).all()
