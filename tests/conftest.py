from pathlib import Path
from typing import Callable

import pytest
from layered_config_tree import LayeredConfigTree
from vivarium.framework.configuration import build_simulation_configuration
from vivarium_testing_utils import FuzzyChecker


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
