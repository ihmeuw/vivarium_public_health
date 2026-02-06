from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def config(base_config):
    base_config.update(
        {
            "population": {
                "initialization_age_min": 0,
                "initialization_age_max": 110,
            },
        },
        source=str(Path(__file__).resolve()),
        layer="model_override",
    )
    return base_config


@pytest.fixture
def generate_population_mock(mocker):
    return mocker.patch(
        "vivarium_public_health.population.base_population.generate_population"
    )


@pytest.fixture
def age_bounds_mock(mocker):
    return mocker.patch(
        "vivarium_public_health.population.base_population._assign_demography_with_age_bounds"
    )


@pytest.fixture
def initial_age_mock(mocker):
    return mocker.patch(
        "vivarium_public_health.population.base_population._assign_demography_with_initial_age"
    )


@pytest.fixture(params=["Male", "Female", "Both"])
def include_sex(request):
    return request.param


@pytest.fixture
def base_simulants():
    simulant_ids = range(100000)
    creation_time = pd.Timestamp(1990, 7, 2)
    return pd.DataFrame(
        {
            "entrance_time": creation_time,
            "exit_time": pd.NaT,
        },
        index=simulant_ids,
    )


@pytest.fixture
def full_simulants(base_simulants):
    base_simulants["location"] = 1
    base_simulants["sex"] = pd.Series("Male", index=base_simulants.index).astype(
        pd.api.types.CategoricalDtype(categories=["Male", "Female"], ordered=False)
    )
    base_simulants["age"] = np.random.uniform(0, 100, len(base_simulants))
    return base_simulants
