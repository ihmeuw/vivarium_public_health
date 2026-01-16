from collections.abc import Callable, Generator
from pathlib import Path

import pytest
from _pytest.logging import LogCaptureFixture
from layered_config_tree import LayeredConfigTree
from loguru import logger
from vivarium.framework.configuration import build_simulation_configuration
from vivarium_testing_utils import FuzzyChecker

from tests.test_utilities import build_table_with_age
from vivarium_public_health.disease import DiseaseModel, DiseaseState
from vivarium_public_health.disease.state import SusceptibleState


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(scope="session")
def base_config_factory() -> Callable[[], LayeredConfigTree]:
    def _base_config() -> LayeredConfigTree:
        config = build_simulation_configuration()

        config.update(
            {
                "time": {"start": {"year": 1990}, "end": {"year": 2010}, "step_size": 30.5},
                "randomness": {"key_columns": ["entrance_time", "age"]},
            },
            source=str(Path(__file__).resolve()),
            layer="model_override",
        )
        return config

    return _base_config


@pytest.fixture(scope="function")
def base_config(base_config_factory) -> LayeredConfigTree:
    yield base_config_factory()


@pytest.fixture(scope="module")
def base_plugins() -> LayeredConfigTree:
    config = {
        "required": {
            "data": {
                "controller": "tests.mock_artifact.MockArtifactManager",
                "builder_interface": "vivarium.framework.artifact.ArtifactInterface",
            }
        }
    }

    return LayeredConfigTree(config)


@pytest.fixture(scope="session")
def fuzzy_checker() -> FuzzyChecker:
    checker = FuzzyChecker()

    yield checker
    test_dir = Path(__file__).resolve().parent
    checker.save_diagnostic_output(test_dir)


@pytest.fixture
def caplog(caplog: LogCaptureFixture) -> Generator[LogCaptureFixture, None, None]:
    handler_id = logger.add(caplog.handler, format="{message}")
    yield caplog
    logger.remove(handler_id)


@pytest.fixture
def disability_disease_models(
    base_config,
    disability_weight_value_0: float,
    disability_weight_value_1: float,
) -> tuple[DiseaseModel, DiseaseModel]:
    """Fixture to create two disease models with disability states for testing.

    Returns a tuple of (model_0, model_1) where each model has a healthy state
    and a sick state with the specified disability weight.
    """
    year_start = base_config.time.start.year
    year_end = base_config.time.end.year

    # Set up two disease models (_0 and _1) to test against multiple causes of disability
    healthy_0 = SusceptibleState("healthy_0")
    healthy_1 = SusceptibleState("healthy_1")

    disability_state_0 = DiseaseState(
        "sick_cause_0",
        disability_weight=build_table_with_age(
            disability_weight_value_0, parameter_columns={"year": (year_start - 1, year_end)}
        ),
        prevalence=build_table_with_age(
            0.45, parameter_columns={"year": (year_start - 1, year_end)}
        ),
    )
    disability_state_1 = DiseaseState(
        "sick_cause_1",
        disability_weight=build_table_with_age(
            disability_weight_value_1, parameter_columns={"year": (year_start - 1, year_end)}
        ),
        prevalence=build_table_with_age(
            0.65, parameter_columns={"year": (year_start - 1, year_end)}
        ),
    )
    # TODO: Add test against using a RiskAttributableDisease in addition to a DiseaseModel
    model_0 = DiseaseModel(
        "model_0", residual_state=healthy_0, states=[healthy_0, disability_state_0]
    )
    model_1 = DiseaseModel(
        "model_1", residual_state=healthy_1, states=[healthy_1, disability_state_1]
    )

    return model_0, model_1
