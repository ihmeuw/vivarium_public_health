from collections import Counter

import numpy as np
import pytest
from vivarium import InteractiveContext
from vivarium.testing_utilities import TestPopulation, build_table

from vivarium_public_health.disease import DiseaseModel, DiseaseState
from vivarium_public_health.disease.state import SusceptibleState
from vivarium_public_health.metrics import MortalityObserver
from vivarium_public_health.metrics.stratification import ResultsStratifier
from vivarium_public_health.population import Mortality


def disease_with_excess_mortality(base_config, disease_name, emr_value) -> DiseaseModel:
    year_start = base_config.time.start.year
    year_end = base_config.time.end.year
    healthy = SusceptibleState(disease_name, allow_self_transition=True)
    disease_get_data_funcs = {
        "disability_weight": lambda *_: build_table(0.0, year_start - 1, year_end),
        "prevalence": lambda *_: build_table(
            0.5, year_start - 1, year_end, ["age", "year", "sex", "value"]
        ),
        "excess_mortality_rate": lambda *_: build_table(
            emr_value, year_start - 1, year_end, ["age", "year", "sex", "value"]
        ),
    }
    with_condition = DiseaseState(disease_name, get_data_functions=disease_get_data_funcs)
    healthy.add_rate_transition(
        with_condition,
        get_data_functions={
            "incidence_rate": lambda *_: build_table(
                0.1, year_start - 1, year_end, ["age", "year", "sex", "value"]
            )
        },
    )
    return DiseaseModel(disease_name, states=[healthy, with_condition])


@pytest.fixture()
def simulation_after_one_step(base_config, base_plugins):
    observer = MortalityObserver()
    flu = disease_with_excess_mortality(base_config, "flu", 10)
    mumps = disease_with_excess_mortality(base_config, "mumps", 20)

    simulation = InteractiveContext(
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
    acmr_data = build_table(0.5, year_start - 1, year_end)
    simulation._data.write("cause.all_causes.cause_specific_mortality_rate", acmr_data)

    simulation.setup()
    simulation.step()

    return simulation


def get_expected_results(simulation, expected_deaths=Counter(), expected_ylls=Counter()):
    """Get expected results given a simulation. If expected deaths, for example, are not provided, return
    the counts of deaths in this time step. If expected deaths are provided, return the counts of deaths
    in the Counter plus the counts for this time step."""
    pop = simulation.get_population()

    for cause in ["other_causes", "flu", "mumps"]:
        for sex in ["Male", "Female"]:
            deaths_observation = f"MEASURE_death_due_to_{cause}_SEX_{sex}"
            ylls_observation = f"MEASURE_ylls_due_to_{cause}_SEX_{sex}"

            died_of_cause = pop["cause_of_death"] == cause
            is_sex_of_interest = pop["sex"] == sex
            died_this_step = pop["exit_time"] == simulation._clock.time
            is_pop_of_interest = died_of_cause & is_sex_of_interest & died_this_step
            expected_deaths.update({deaths_observation: sum(is_pop_of_interest)})
            expected_ylls.update(
                {ylls_observation: sum(pop.loc[is_pop_of_interest, "years_of_life_lost"])}
            )

    return expected_deaths, expected_ylls


def test_observation_registration(simulation_after_one_step):
    """Test that all expected observation keys appear as expected in the results."""
    results = simulation_after_one_step.get_value("metrics")
    pop = simulation_after_one_step.get_population()

    expected_observations = [
        "MEASURE_death_due_to_other_causes_SEX_Female",
        "MEASURE_death_due_to_other_causes_SEX_Male",
        "MEASURE_ylls_due_to_other_causes_SEX_Female",
        "MEASURE_ylls_due_to_other_causes_SEX_Male",
        "MEASURE_death_due_to_flu_SEX_Female",
        "MEASURE_death_due_to_flu_SEX_Male",
        "MEASURE_ylls_due_to_flu_SEX_Female",
        "MEASURE_ylls_due_to_flu_SEX_Male",
        "MEASURE_death_due_to_mumps_SEX_Female",
        "MEASURE_death_due_to_mumps_SEX_Male",
        "MEASURE_ylls_due_to_mumps_SEX_Female",
        "MEASURE_ylls_due_to_mumps_SEX_Male",
    ]

    assert set(expected_observations) == set(results(pop.index).keys())


def test_observation_correctness(simulation_after_one_step):
    """Test that deaths and YLLs appear as expected in the results."""
    expected_deaths, expected_ylls = get_expected_results(simulation_after_one_step)
    results = simulation_after_one_step.get_value("metrics")
    pop = simulation_after_one_step.get_population()

    for observation in expected_deaths:
        assert np.isclose(
            results(pop.index)[observation], expected_deaths[observation], rtol=0.001
        )
    for observation in expected_ylls:
        assert np.isclose(
            results(pop.index)[observation], expected_ylls[observation], rtol=0.001
        )

    # same test on second time step
    simulation_after_one_step.step()
    expected_deaths, expected_ylls = get_expected_results(
        simulation_after_one_step,
        expected_deaths=expected_deaths,
        expected_ylls=expected_ylls,
    )
    pop = simulation_after_one_step.get_population()
    results = simulation_after_one_step.get_value("metrics")

    for observation in expected_deaths:
        assert np.isclose(
            results(pop.index)[observation], expected_deaths[observation], rtol=0.001
        )
    for observation in expected_ylls:
        assert np.isclose(
            results(pop.index)[observation], expected_ylls[observation], rtol=0.001
        )


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
    acmr_data = build_table(0.5, year_start - 1, year_end)
    aggregate_sim._data.write("cause.all_causes.cause_specific_mortality_rate", acmr_data)
    aggregate_sim.setup()
    aggregate_sim.step()
    results = aggregate_sim.get_value("metrics")
    pop = aggregate_sim.get_population()

    expected_observations = [
        "MEASURE_death_due_to_all_causes_SEX_Female",
        "MEASURE_death_due_to_all_causes_SEX_Male",
        "MEASURE_ylls_due_to_all_causes_SEX_Female",
        "MEASURE_ylls_due_to_all_causes_SEX_Male",
    ]

    assert set(expected_observations) == set(results(pop.index).keys())
