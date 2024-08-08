import itertools

import numpy as np
import pandas as pd
import pytest
from vivarium import InteractiveContext
from vivarium.testing_utilities import TestPopulation

from tests.test_utilities import build_table_with_age
from vivarium_public_health.disease import DiseaseModel, DiseaseState
from vivarium_public_health.disease.state import SusceptibleState
from vivarium_public_health.results.columns import COLUMNS
from vivarium_public_health.results.disease import DiseaseObserver
from vivarium_public_health.results.stratification import ResultsStratifier
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


@pytest.fixture
def vampiris():
    vampiris_healthy_state = SusceptibleState("human")
    vampiris_turning_state = DiseaseState("turning")
    vampiris_infected_state = DiseaseState("vampire")
    vampiris_healthy_state.add_rate_transition(vampiris_turning_state)
    vampiris_turning_state.add_rate_transition(vampiris_infected_state)
    return DiseaseModel(
        "vampiris",
        initial_state=vampiris_healthy_state,
        states=[vampiris_healthy_state, vampiris_turning_state, vampiris_infected_state],
    )


@pytest.fixture
def human_cortico_deficiency():
    hcd_healthy_state = SusceptibleState("not_a_zombie")
    hcd_infected_state = DiseaseState("a_zombie")
    hcd_healthy_state.add_rate_transition(hcd_infected_state)
    return DiseaseModel(
        "human_cortico_deficiency",
        initial_state=hcd_healthy_state,
        states=[hcd_healthy_state, hcd_infected_state],
    )


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

    # Assert that the previous_state column equals the current state column
    assert (pop[observer.previous_state_column_name] == pop[observer.disease]).all()

    simulation.step()
    post_step_pop = simulation.get_population()

    # All simulants are currently but not necessarily previously "with_condition"
    assert (
        post_step_pop[observer.previous_state_column_name].isin(
            ["susceptible_to_with_condition", "with_condition"]
        )
    ).all()
    assert (post_step_pop[observer.disease] == "with_condition").all()

    simulation.step()
    post_step_pop = simulation.get_population()

    # All simulants are currently and were previously "with_condition"
    assert (post_step_pop[observer.previous_state_column_name] == "with_condition").all()
    assert (post_step_pop[observer.disease] == "with_condition").all()


def test_observation_registration(base_config, base_plugins, disease, model):
    """Test that all expected observation stratifications appear in the results."""
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
    simulation.step()
    results = simulation.get_results()
    person_time = results["person_time_t_virus"]
    transition_count = results["transition_count_t_virus"]

    # Check that all expected observations are present
    assert set(zip(person_time[COLUMNS.SUB_ENTITY], person_time["sex"])) == set(
        itertools.product(
            *[["susceptible_to_with_condition", "with_condition"], ["Female", "Male"]]
        )
    )
    assert set(zip(transition_count[COLUMNS.SUB_ENTITY], transition_count["sex"])) == set(
        itertools.product(
            *[["susceptible_to_with_condition_to_with_condition"], ["Female", "Male"]]
        )
    )


# Person time and all states and transition counts are correct
def test_observation_correctness(base_config, base_plugins, disease, model):
    """Test that person time and event counts appear as expected in the results."""
    time_step = pd.Timedelta(days=base_config.time.step_size)
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

    # All simulants should transition to "with_condition"
    susceptible_at_start = len(pop[pop[disease] == "susceptible_to_with_condition"])
    expected_susceptible_person_time = susceptible_at_start * to_years(time_step)
    expected_with_condition_person_time = (len(pop) - susceptible_at_start) * to_years(
        time_step
    )

    simulation.step()
    results = simulation.get_results()
    person_time = results["person_time_t_virus"]
    transition_count = results["transition_count_t_virus"]

    # Check columns
    for measure in ["person_time", "transition_count"]:
        df = eval(measure)
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
        assert (df[COLUMNS.ENTITY] == "t_virus").all()

    # Check values
    actual_tx_count = transition_count.loc[
        transition_count[COLUMNS.SUB_ENTITY]
        == "susceptible_to_with_condition_to_with_condition",
        COLUMNS.VALUE,
    ].sum()
    actual_person_times = person_time.groupby(COLUMNS.SUB_ENTITY)[COLUMNS.VALUE].sum()
    assert np.isclose(actual_tx_count, susceptible_at_start, rtol=0.001)
    assert np.isclose(
        actual_person_times["susceptible_to_with_condition"],
        expected_susceptible_person_time,
        rtol=0.001,
    )
    assert np.isclose(
        actual_person_times["with_condition"], expected_with_condition_person_time, rtol=0.001
    )


def test_different_results_per_disease(
    vampiris, human_cortico_deficiency, base_config, base_plugins
):
    """Test that all eash disease observer saves out its own results."""
    vampiris_observer = DiseaseObserver(vampiris.cause)
    hcd_observer = DiseaseObserver(human_cortico_deficiency.cause)

    simulation = InteractiveContext(
        components=[
            TestPopulation(),
            vampiris,
            human_cortico_deficiency,
            ResultsStratifier(),
            vampiris_observer,
            hcd_observer,
        ],
        configuration=base_config,
        plugin_configuration=base_plugins,
        setup=False,
    )

    simulation.setup()
    simulation.step()
    results = simulation.get_results()
    assert set(results) == set(
        [
            "person_time_vampiris",
            "transition_count_vampiris",
            "person_time_human_cortico_deficiency",
            "transition_count_human_cortico_deficiency",
        ]
    )


@pytest.mark.parametrize(
    "person_time_exclusions, transition_count_exclusions",
    [
        ([], []),
        (["susceptible_to_human"], ["susceptible_to_human_to_turning"]),
        (["susceptible_to_human", "turning"], []),
    ],
)
def test_category_exclusions(
    vampiris, base_config, base_plugins, person_time_exclusions, transition_count_exclusions
):
    """Test that we can exclude diseases via the model spec."""
    vampiris_observer = DiseaseObserver(vampiris.cause)

    # Add exclusions to model spec
    base_config.update(
        {
            "stratification": {
                "excluded_categories": {
                    "vampiris": person_time_exclusions,
                    "transition_vampiris": transition_count_exclusions,
                }
            }
        }
    )
    simulation = InteractiveContext(
        components=[
            TestPopulation(),
            vampiris,
            ResultsStratifier(),
            vampiris_observer,
        ],
        configuration=base_config,
        plugin_configuration=base_plugins,
    )

    simulation.step()
    person_time = simulation.get_results()["person_time_vampiris"]
    transition_count = simulation.get_results()["transition_count_vampiris"]
    assert set(person_time["sub_entity"]) == set(vampiris.state_names) - set(
        person_time_exclusions
    )
    assert set(transition_count["sub_entity"]) == set(vampiris.transition_names) - set(
        transition_count_exclusions
    )
