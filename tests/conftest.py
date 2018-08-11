import pytest

from vivarium.framework.configuration import build_simulation_configuration
from vivarium.testing_utilities import metadata
from vivarium.config_tree import ConfigTree


@pytest.fixture(scope='module')
def base_config():
    config = build_simulation_configuration()

    config.update({
        'time': {
            'start': {'year': 1990},
            'end': {'year': 2010},
            'step_size': 30.5
        },
        'randomness': {'key_columns': ['entrance_time', 'age']},
        'input_data': {'location': 'Kenya'},
    }, **metadata(__file__))

    return config


@pytest.fixture(scope='module')
def base_plugins():
    config = {'optional': {
                  'data': {
                      'controller': 'vivarium_public_health.testing.mock_artifact.MockArtifactManager',
                      'builder_interface': 'vivarium_public_health.dataset_manager.ArtifactManagerInterface'
                  }
             }
    }

    return ConfigTree(config)
