import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from vivarium import InteractiveContext
from vivarium.testing_utilities import TestPopulation, build_table

from tests.test_utilities import build_table_with_age
from vivarium_public_health.disease import DiseaseModel, DiseaseState
from vivarium_public_health.disease.state import SusceptibleState
from vivarium_public_health.metrics.disease import DiseaseObserver
from vivarium_public_health.metrics.reporters import COLUMNS
from vivarium_public_health.metrics.stratification import ResultsStratifier
from vivarium_public_health.utilities import to_years


@pytest.fixture
def disease() -> str:
    return "t_virus"


@pytest.fixture
def model(base_config, disease: str) -> DiseaseModel:
    """A dummy SI model where everyone should be `with_condition` by the third timestep."""
    year_start = base_config.time.start.year
    year_end = base_config.time.end.year
    healthy = SusceptibleState("with_condition")
    disease_get_data_funcs = {
        "disability_weight": lambda _, __: build_table_with_age(
            0.0, parameter_columns={"year": (year_start - 1, year_end)}
        ),
        "prevalence": lambda _, __: build_table_with_age(
            0.2, parameter_columns={"year": (year_start - 1, year_end)}
        ),
    }
    transition_get_data_funcs = {
        "incidence_rate": lambda _, __: build_table_with_age(
            0.9, parameter_columns={"year": (year_start - 1, year_end)}
        ),
    }
    with_condition = DiseaseState("with_condition", get_data_functions=disease_get_data_funcs)
    healthy.add_rate_transition(with_condition, transition_get_data_funcs)
    return DiseaseModel(disease, initial_state=healthy, states=[healthy, with_condition])


# Updating the previous state
def test_previous_state_update(base_config, base_plugins, disease, model):
    """Test that the observer previous_state column is updated as expected."""
    observer = DiseaseObserver(disease)
    simulation = InteractiveContext(
        components=[
            TestPopulation(),
            model,
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
                "t_virus": {
                    "include": ["sex"],
                }
            }
        }
    )

    simulation.setup()

    pop = simulation.get_population()

    # Assert that the previous_state column is all empty
    assert (pop[observer.previous_state_column_name] == "").all()

    simulation.step()
    post_step_pop = simulation.get_population()

    # All simulants are currently but not necessarily previously "with_condition"
    assert (
        post_step_pop[observer.previous_state_column_name].isin(
            ["susceptible_to_with_condition", "with_condition"]
        )
    ).all()
    assert (post_step_pop[observer.current_state_column_name] == "with_condition").all()

    simulation.step()
    post_step_pop = simulation.get_population()

    # All simulants are currently and were previously "with_condition"
    assert (post_step_pop[observer.previous_state_column_name] == "with_condition").all()
    assert (post_step_pop[observer.current_state_column_name] == "with_condition").all()


def test_observation_registration(base_config, base_plugins, disease, model, tmpdir):
    """Test that all expected observation stratifications appear in the results."""
    observer = DiseaseObserver(disease)
    # Add the results dir since we didn't go through cli.py
    results_dir = Path(tmpdir)
    base_config.update({"output_data": {"results_directory": str(results_dir)}})
    simulation = InteractiveContext(
        components=[
            TestPopulation(),
            model,
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
                "t_virus": {
                    "include": ["sex"],
                }
            }
        }
    )

    simulation.setup()
    simulation.step()
    simulation.finalize()
    simulation.report()
    results_files = list(results_dir.rglob("*.parquet"))
    assert set(file.name for file in results_files) == set(
        ["state_person_time.parquet", "transition_count.parquet"]
    )
    state_person_time = pd.read_parquet(results_dir / "state_person_time.parquet")
    transition_count = pd.read_parquet(results_dir / "transition_count.parquet")

    # Check that all expected observations are present
    assert set(zip(state_person_time["state"], state_person_time["sex"])) == set(
        itertools.product(
            *[["susceptible_to_with_condition", "with_condition"], ["Female", "Male"]]
        )
    )
    assert set(zip(transition_count["transition"], transition_count["sex"])) == set(
        itertools.product(
            *[["susceptible_to_with_condition_to_with_condition"], ["Female", "Male"]]
        )
    )


# Person time and all states and transition counts are correct
def test_observation_correctness(base_config, base_plugins, disease, model, tmpdir):
    """Test that person time and event counts appear as expected in the results."""
    time_step = pd.Timedelta(days=base_config.time.step_size)
    observer = DiseaseObserver(disease)
    # Add the results dir since we didn't go through cli.py
    results_dir = Path(tmpdir)
    base_config.update({"output_data": {"results_directory": str(results_dir)}})
    simulation = InteractiveContext(
        components=[
            TestPopulation(),
            model,
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
                "t_virus": {
                    "include": ["sex"],
                }
            }
        }
    )

    simulation.setup()
    pop = simulation.get_population()

    # All simulants should transition to "with_condition"
    susceptible_at_start = len(pop[pop[disease] == "susceptible_to_with_condition"])
    expected_susceptible_person_time = susceptible_at_start * to_years(time_step)
    expected_with_condition_person_time = (len(pop) - susceptible_at_start) * to_years(
        time_step
    )

    simulation.step()
    simulation.finalize()
    simulation.report()

    state_person_time = pd.read_parquet(results_dir / "state_person_time.parquet")
    transition_count = pd.read_parquet(results_dir / "transition_count.parquet")

    # Check columns (NOTE: no input_draw defined so shouldn't be there)
    assert set(state_person_time.columns) == set(
        ["sex", COLUMNS.STATE, COLUMNS.MEASURE, COLUMNS.SEED, COLUMNS.VALUE]
    )
    assert (state_person_time[COLUMNS.MEASURE] == "state_person_time").all()
    assert (state_person_time[COLUMNS.SEED] == 0).all()
    assert set(transition_count.columns) == set(
        ["sex", COLUMNS.TRANSITION, COLUMNS.MEASURE, COLUMNS.SEED, COLUMNS.VALUE]
    )
    assert (transition_count[COLUMNS.MEASURE] == "transition_count").all()
    assert (transition_count[COLUMNS.SEED] == 0).all()

    # Check values
    actual_tx_count = transition_count.loc[
        transition_count[COLUMNS.TRANSITION]
        == "susceptible_to_with_condition_to_with_condition",
        COLUMNS.VALUE,
    ].sum()
    actual_person_times = state_person_time.groupby("state")[COLUMNS.VALUE].sum()
    assert np.isclose(actual_tx_count, susceptible_at_start, rtol=0.001)
    assert np.isclose(
        actual_person_times["susceptible_to_with_condition"],
        expected_susceptible_person_time,
        rtol=0.001,
    )
    assert np.isclose(
        actual_person_times["with_condition"], expected_with_condition_person_time, rtol=0.001
    )
