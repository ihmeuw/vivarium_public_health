import pytest
from vivarium_public_health.dataset_manager import Artifact


def test_artifact_load_empty_data(mocker):
    uncached_load_mock = mocker.patch('vivarium_public_health.dataset_manager.Artifact._uncached_load')
    uncached_load_mock.return_value = None

    test_artifact = Artifact('test_path', 2010, 2012, 0, 'test_location')
    with pytest.raises(Exception) as e:
        test_artifact.load('test_entity_path')
    assert e.typename == 'ArtifactException'
    assert e.value.args[0] == 'data for test_entity_path is not available. Check your model specification'

