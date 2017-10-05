import pytest

from vivarium.framework.engine import build_simulation_configuration


@pytest.fixture(scope='module')
def base_config():
    return build_simulation_configuration({})
