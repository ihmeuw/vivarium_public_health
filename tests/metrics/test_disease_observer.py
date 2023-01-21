from collections import namedtuple

import numpy as np
import pandas as pd
import pytest
from vivarium import InteractiveContext
from vivarium.testing_utilities import TestPopulation, build_table

from vivarium_public_health.disease import (
    DiseaseModel,
    DiseaseState,
    RiskAttributableDisease,
)
from vivarium_public_health.disease.state import SusceptibleState
from vivarium_public_health.disease.transition import TransitionString
from vivarium_public_health.metrics.disease import (
    DiseaseObserver as DiseaseObserver_,
)
from vivarium_public_health.metrics.stratification import (
    ResultsStratifier as ResultsStratifier_,
)


# Subclass of ResultsStratifier for integration testing
class ResultsStratifier(ResultsStratifier_):
    configuration_defaults = {
        "stratification": {
            "default": ["age_group", "sex"],
        }
    }


# Subclass of DisabilityObserver for integration testing
class DiseaseObserver(DiseaseObserver_):
    configuration_defaults = {
        "stratification": {
            "disease": {
                "exclude": ["age_group"],
                "include": ["sex"],
            }
        }
    }


# Updating the previous state
def test_previous_state_update(base_config, base_plugins):
    # Setup disease and transition
    year_start = base_config.time.start.year
    year_end = base_config.time.end.year
    time_step = pd.Timedelta(days=base_config.time.step_size)
    healthy = SusceptibleState("healthy")
    disease_get_data_funcs = {
        "disability_weight": lambda _, __: build_table(
            0.0, year_start - 1, year_end
        ),
        "prevalence": lambda _, __: build_table(
            0.2, year_start - 1, year_end, ["age", "year", "sex", "value"]
        ),
        "incidence": lambda _, __: build_table(
            0.9, year_start - 1, year_end, ["age", "year", "sex", "value"]
        ),
    }
    with_condition = DiseaseState(
        "with_condition", get_data_functions=disease_get_data_funcs
    )
    healthy.add_transition(with_condition)
    model = DiseaseModel(
        "disease", initial_state=healthy, states=[healthy, with_condition]
    )
    observer = DiseaseObserver("disease")
    simulation = InteractiveContext(
        components=[
            TestPopulation(),
            model,
            ResultsStratifier(),
            observer,
        ],
        configuration=base_config,
        plugin_configuration=base_plugins,
    )
    pop = simulation.get_population()

    # Assert that the previous_state column is all ""
    assert (pop[observer.previous_state_column_name] == '').all()

    simulation.step()


    assert True


# Observations are registered for person time and all states and transition counts
def test_person_time():
    assert True


# Person time and all states and transition counts are correct
def test_transition_counts():
    assert True

