import pytest

from vivarium.framework.configuration import build_simulation_configuration
from vivarium.test_util import metadata


@pytest.fixture(scope='module')
def base_config():
    config = build_simulation_configuration()

    config.update({
        'time': {
            'start': {'year': 1990},
            'end': {'year': 2010},
            'step_size': 30.5
        },
        'randomness': {'key_columns': ['entrance_time', 'age']}
    }, **metadata(__file__))

    return config
