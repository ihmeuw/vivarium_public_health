"""Tests for lifecycle event ordering.

These tests verify the desired ordering of population lifecycle events:
    fertility (prepare) → age up → observe person-time → mortality → other observations (collect_metrics)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from vivarium import InteractiveContext

from tests.test_utilities import build_table_with_age
from vivarium_public_health.disease import DiseaseModel, DiseaseState
from vivarium_public_health.disease.state import SusceptibleState
from vivarium_public_health.population import BasePopulation, FertilityDeterministic
from vivarium_public_health.results.columns import COLUMNS
from vivarium_public_health.results.disease import DiseaseObserver
from vivarium_public_health.results.stratification import ResultsStratifier
from vivarium_public_health.utilities import to_years


@pytest.fixture
def disease_model(base_config):
    """A simple SI disease model with high excess mortality rate."""
    year_start = base_config.time.start.year
    year_end = base_config.time.end.year
    healthy = SusceptibleState("sick")
    sick = DiseaseState(
        "sick",
        disability_weight=build_table_with_age(
            0.0, parameter_columns={"year": (year_start - 1, year_end)}
        ),
        prevalence=build_table_with_age(
            0.5, parameter_columns={"year": (year_start - 1, year_end)}
        ),
        excess_mortality_rate=build_table_with_age(
            10, parameter_columns={"year": (year_start - 1, year_end)}
        ),
    )
    healthy.add_rate_transition(
        sick,
        transition_rate=build_table_with_age(
            0.5, parameter_columns={"year": (year_start - 1, year_end)}
        ),
    )
    return DiseaseModel("test_disease", states=[healthy, sick])


@pytest.mark.xfail(reason="New lifecycle ordering not yet implemented", strict=True)
def test_fertility_before_aging(base_config, base_plugins):
    """Test that fertility fires before aging so newborns are aged during the same step.

    Under the new ordering:
    - Fertility fires in on_time_step_prepare, creating newborns at age=0
    - Aging fires in on_time_step, incrementing all simulant ages (including newborns)

    Therefore, after one time step, newborns should have age > 0 (specifically
    age == step_size_in_years).
    """
    step_size_days = 30.5
    base_config.update(
        {
            "population": {
                "population_size": 100,
                "initialization_age_min": 0,
                "initialization_age_max": 125,
            },
            "time": {"step_size": step_size_days},
            "fertility": {"number_of_new_simulants_each_year": 3650},
        },
        source=str(Path(__file__).resolve()),
        layer="override",
    )

    simulation = InteractiveContext(
        components=[BasePopulation(), FertilityDeterministic()],
        configuration=base_config,
        plugin_configuration=base_plugins,
    )

    initial_pop_size = len(simulation.get_population())
    simulation.step()
    pop = simulation.get_population(["age", "entrance_time"])

    # Newborns should exist
    newborns = pop[pop["entrance_time"] > simulation._clock.time - simulation._clock.step_size]
    assert len(newborns) > 0, "No newborns were created"

    # Under the new ordering, newborns should have been aged (age > 0)
    # because fertility fires in prepare and aging fires in time_step
    step_size_years = to_years(pd.Timedelta(days=step_size_days))
    assert np.all(
        newborns["age"] > 0
    ), "Newborns were not aged - fertility must fire before aging"
    assert np.allclose(
        newborns["age"], step_size_years, atol=step_size_years * 0.1
    ), "Newborn ages don't match expected step size"


@pytest.mark.xfail(reason="New lifecycle ordering not yet implemented", strict=True)
def test_person_time_includes_dead_simulants(base_config, base_plugins, disease_model):
    """Test that person-time observation fires before mortality.

    Under the new ordering:
    - Person-time observation fires during on_time_step at priority 5
    - Mortality fires during on_time_step at priority 6

    Therefore, simulants who die this step should still contribute person-time
    because person-time was counted while they were still alive.
    """
    base_config.update(
        {
            "population": {
                "population_size": 1000,
                "initialization_age_min": 0,
                "initialization_age_max": 125,
            },
        },
        source=str(Path(__file__).resolve()),
        layer="override",
    )

    disease_name = "test_disease"
    observer = DiseaseObserver(disease_name)
    simulation = InteractiveContext(
        components=[
            BasePopulation(),
            disease_model,
            ResultsStratifier(),
            observer,
        ],
        configuration=base_config,
        plugin_configuration=base_plugins,
        setup=False,
    )
    simulation.configuration.update(
        {"stratification": {disease_name: {"include": []}}}
    )

    year_start = base_config.time.start.year
    year_end = base_config.time.end.year
    acmr_data = build_table_with_age(
        0.5, parameter_columns={"year": (year_start - 1, year_end)}
    )
    simulation._data.write("cause.all_causes.cause_specific_mortality_rate", acmr_data)

    simulation.setup()

    # Get initial population size (all alive)
    pop_before = simulation.get_population(["is_alive"])
    initial_alive_count = pop_before["is_alive"].sum()

    simulation.step()

    # Get post-step population state
    pop_after = simulation.get_population(["is_alive"])
    deaths_this_step = initial_alive_count - pop_after["is_alive"].sum()
    assert deaths_this_step > 0, "No deaths occurred - test is not meaningful"

    # Get person-time results
    results = simulation.get_results()
    person_time = results[f"person_time_{disease_name}"]
    total_person_time = person_time[COLUMNS.VALUE].sum()

    # Under the new ordering, person-time is observed BEFORE mortality.
    # So ALL initially-alive simulants should contribute person-time,
    # including those who die this step.
    time_step = pd.Timedelta(days=base_config.time.step_size)
    expected_person_time = initial_alive_count * to_years(time_step)

    assert np.isclose(total_person_time, expected_person_time, rtol=0.01), (
        f"Person-time ({total_person_time:.4f}) does not equal expected "
        f"({expected_person_time:.4f}). Under the new ordering, all initially-alive "
        f"simulants (including {deaths_this_step} who died) should contribute person-time "
        f"because observation fires before mortality."
    )


@pytest.mark.xfail(reason="New lifecycle ordering not yet implemented", strict=True)
def test_aging_before_person_time_observation(base_config, base_plugins, disease_model):
    """Test that aging fires before person-time observation.

    Under the new ordering:
    - Aging fires during on_time_step at priority 2
    - Person-time observation fires during on_time_step at priority 5 (via ResultsManager)

    This test verifies the ordering by confirming that person-time observation
    fires during on_time_step (not during time_step__prepare as it does currently).
    We check this indirectly: if person-time fires after aging and before mortality,
    then the total person-time should equal the count of all alive simulants
    (post-aging, pre-mortality) times the step size. Combined with
    test_person_time_includes_dead_simulants, this confirms the full ordering.
    """
    base_config.update(
        {
            "population": {
                "population_size": 1000,
                "initialization_age_min": 0,
                "initialization_age_max": 125,
            },
            "fertility": {"number_of_new_simulants_each_year": 3650},
        },
        source=str(Path(__file__).resolve()),
        layer="override",
    )

    disease_name = "test_disease"
    observer = DiseaseObserver(disease_name)
    simulation = InteractiveContext(
        components=[
            BasePopulation(),
            FertilityDeterministic(),
            disease_model,
            ResultsStratifier(),
            observer,
        ],
        configuration=base_config,
        plugin_configuration=base_plugins,
        setup=False,
    )
    simulation.configuration.update(
        {"stratification": {disease_name: {"include": []}}}
    )

    year_start = base_config.time.start.year
    year_end = base_config.time.end.year
    acmr_data = build_table_with_age(
        0.5, parameter_columns={"year": (year_start - 1, year_end)}
    )
    simulation._data.write("cause.all_causes.cause_specific_mortality_rate", acmr_data)

    simulation.setup()

    initial_pop_size = len(simulation.get_population())

    simulation.step()

    pop_after = simulation.get_population(["is_alive", "entrance_time"])
    # Newborns are those created during the time step
    newborns = pop_after[
        pop_after["entrance_time"] > simulation._clock.time - simulation._clock.step_size
    ]
    total_alive_at_observation = initial_pop_size + len(newborns)

    # Person-time should reflect ALL simulants that were alive at observation time
    # (after fertility + aging, before mortality)
    results = simulation.get_results()
    person_time = results[f"person_time_{disease_name}"]
    total_person_time = person_time[COLUMNS.VALUE].sum()

    time_step = pd.Timedelta(days=base_config.time.step_size)
    expected_person_time = total_alive_at_observation * to_years(time_step)

    assert np.isclose(total_person_time, expected_person_time, rtol=0.01), (
        f"Person-time ({total_person_time:.4f}) does not match expected "
        f"({expected_person_time:.4f}). Expected person-time to include "
        f"{initial_pop_size} initial simulants + {len(newborns)} newborns "
        f"(fertility fires in prepare, so newborns exist at observation time)."
    )
