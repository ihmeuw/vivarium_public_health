from pathlib import Path

import pytest
from vivarium.framework.configuration import build_simulation_configuration
from vivarium.config_tree import ConfigTree


@pytest.fixture()
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
    }, source=str(Path(__file__).resolve()), layer='model_override')

    return config


@pytest.fixture(scope='module')
def base_plugins():
    config = {'required': {
                  'data': {
                      'controller': 'vivarium_public_health.testing.mock_artifact.MockArtifactManager',
                      'builder_interface': 'vivarium.framework.artifact.ArtifactInterface'
                  }
             }
    }

    return ConfigTree(config)
