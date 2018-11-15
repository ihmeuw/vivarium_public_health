import pytest
import pandas as pd
from pathlib import Path

from vivarium_public_health.dataset_manager.artifact import Artifact, ArtifactException, EntityKey, _to_tree


@pytest.fixture()
def keys_mock():
    keys = ['cause.all_causes.cause_specific_mortality', 'cause.all_causes.restrictions',
            'cause.diarrheal_diseases.cause_specific_mortality', 'cause.diarrheal_diseases.death',
            'cause.diarrheal_diseases.etiologies', 'cause.diarrheal_diseases.excess_mortality',
            'cause.diarrheal_diseases.incidence', 'cause.diarrheal_diseases.population_attributable_fraction',
            'cause.diarrheal_diseases.prevalence', 'cause.diarrheal_diseases.remission',
            'cause.diarrheal_diseases.restrictions', 'cause.diarrheal_diseases.sequelae',
            'covariate.dtp3_coverage_proportion.estimate', 'dimensions.full_space',
            'etiology.adenovirus.population_attributable_fraction',
            'etiology.aeromonas.population_attributable_fraction',
            'etiology.amoebiasis.population_attributable_fraction', 'population.age_bins', 'population.structure',
            'population.theoretical_minimum_risk_life_expectancy', 'risk_factor.child_stunting.affected_causes',
            'risk_factor.child_stunting.distribution', 'risk_factor.child_stunting.exposure',
            'risk_factor.child_stunting.levels', 'risk_factor.child_stunting.relative_risk',
            'risk_factor.child_stunting.restrictions', 'sequela.moderate_diarrheal_diseases.disability_weight',
            'sequela.moderate_diarrheal_diseases.healthstate', 'sequela.moderate_diarrheal_diseases.incidence',
            'sequela.moderate_diarrheal_diseases.prevalence', 'no_data.key']
    return keys


@pytest.fixture()
def hdf_mock(mocker, keys_mock):
    mock = mocker.patch('vivarium_public_health.dataset_manager.artifact.hdf')
    mock.get_keys.return_value = keys_mock

    def mock_load(_, key, __):
        if str(key) in keys_mock and key != 'no_data.key':
            return 'data'
        else:
            return None

    mock.load.side_effect = mock_load

    return mock

# keys in test artifact
_KEYS = ['population.age_bins',
         'population.structure',
         'population.theoretical_minimum_risk_life_expectancy',
         'cause.all_causes.restrictions']


def test_artifact_creation(hdf_mock, keys_mock):
    path = '/place/with/artifact.hdf'
    filter_terms = ['location == Global', 'draw == 10']

    a = Artifact(path)

    assert a.path == path
    assert a.filter_terms is None
    assert a._cache == {}
    assert a.keys == [EntityKey(k) for k in keys_mock]
    hdf_mock.get_keys.assert_called_once_with(path)

    hdf_mock.get_keys.reset_mock()

    a = Artifact(path, filter_terms)

    assert a.path == path
    assert a.filter_terms == filter_terms
    assert a._cache == {}
    assert a.keys == [EntityKey(k) for k in keys_mock]
    hdf_mock.get_keys.assert_called_once_with(path)


def test_artifact_load_missing_key(hdf_mock):
    path = '/place/with/artifact.hdf'
    filter_terms = ['location == Global', 'draw == 10']
    key = 'not.a_real.key'

    a = Artifact(path, filter_terms)

    with pytest.raises(ArtifactException) as err_info:
        a.load(key)

    assert f"{key} should be in {path}." == str(err_info.value)
    hdf_mock.load.assert_not_called()
    assert a._cache == {}


def test_artifact_load_key_has_no_data(hdf_mock):
    path = '/place/with/artifact.hdf'
    filter_terms = ['location == Global', 'draw == 10']
    key = 'no_data.key'
    ekey = EntityKey(key)

    a = Artifact(path, filter_terms)

    with pytest.raises(AssertionError) as err_info:
        a.load(key)

    assert f"Data for {key} is not available. Check your model specification." == str(err_info.value)
    assert hdf_mock.load.called_once_with(path, ekey, filter_terms)
    assert a._cache == {}


def test_artifact_load(hdf_mock, keys_mock):
    path = '/place/with/artifact.hdf'
    filter_terms = ['location == Global', 'draw == 10']

    a = Artifact(path, filter_terms)

    for key in keys_mock:
        ekey = EntityKey(key)
        if key == 'no_data.key':
            continue

        assert ekey not in a._cache

        result = a.load(key)

        assert hdf_mock.load.called_once_with(path, ekey, filter_terms)
        assert ekey in a._cache
        assert a._cache[ekey] == 'data'
        assert result == 'data'

        hdf_mock.load.reset_mock()


def test_artifact_write_duplicate_key(hdf_mock):
    path = '/place/with/artifact.hdf'
    filter_terms = ['location == Global', 'draw == 10']
    key = 'population.structure'
    ekey = EntityKey(key)

    a = Artifact(path, filter_terms)

    with pytest.raises(ArtifactException) as err_info:
        a.write(key, "data")

    assert f'{key} already in artifact.' == str(err_info.value)
    assert key in a
    assert ekey in a.keys
    assert ekey not in a._cache
    hdf_mock.write.assert_not_called()


def test_artifact_write_no_data(hdf_mock):
    path = '/place/with/artifact.hdf'
    filter_terms = ['location == Global', 'draw == 10']
    key = 'new.key'
    ekey = EntityKey(key)

    a = Artifact(path, filter_terms)

    assert ekey not in a.keys

    a.write(key, None)

    assert ekey not in a.keys
    assert ekey not in a._cache
    hdf_mock.write.assert_not_called()


def test_artifact_write(hdf_mock):
    path = '/place/with/artifact.hdf'
    filter_terms = ['location == Global', 'draw == 10']
    key = 'new.key'
    ekey = EntityKey(key)

    a = Artifact(path, filter_terms)

    assert ekey not in a.keys

    a.write(key, "data")

    assert ekey in a.keys
    assert ekey not in a._cache
    hdf_mock.write.assert_called_once_with(path, ekey, "data")


def test_remove_bad_key(hdf_mock):
    path = '/place/with/artifact.hdf'
    filter_terms = ['location == Global', 'draw == 10']
    key = 'non_existent.key'
    ekey = EntityKey(key)

    a = Artifact(path, filter_terms)

    assert ekey not in a.keys
    assert ekey not in a._cache

    with pytest.raises(ArtifactException) as err_info:
        a.remove(key)

    assert f'Trying to remove non-existent key {key} from artifact.' == str(err_info.value)
    assert ekey not in a.keys
    assert ekey not in a._cache
    hdf_mock.remove.assert_not_called()


def test_remove_no_cache(hdf_mock):
    path = '/place/with/artifact.hdf'
    filter_terms = ['location == Global', 'draw == 10']
    key = 'population.structure'
    ekey = EntityKey(key)

    a = Artifact(path, filter_terms)

    assert ekey in a.keys
    assert ekey not in a._cache

    a.remove(key)

    assert ekey not in a.keys
    assert ekey not in a._cache
    hdf_mock.remove.assert_called_once_with(path, ekey)


def test_remove(hdf_mock):
    path = '/place/with/artifact.hdf'
    filter_terms = ['location == Global', 'draw == 10']
    key = 'population.structure'
    ekey = EntityKey(key)

    a = Artifact(path, filter_terms)
    a._cache[ekey] = 'data'

    assert ekey in a.keys
    assert ekey in a._cache

    a.remove(key)

    assert ekey not in a.keys
    assert ekey not in a._cache
    hdf_mock.remove.assert_called_once_with(path, ekey)


def test_clear_cache(hdf_mock):
    path = '/place/with/artifact.hdf'
    filter_terms = ['location == Global', 'draw == 10']
    key = 'population.structure'
    ekey = EntityKey(key)

    a = Artifact(path, filter_terms)
    a.clear_cache()

    assert a._cache == {}

    a._cache[ekey] = 'data'
    a.clear_cache()

    assert a._cache == {}


def test_loading_key_leaves_filters_unchanged():
    # loading each key will drop the fake_filter from filter_terms for that key
    # make sure that artifact's filter terms stay the same though
    path = str(Path(__file__).parent / 'artifact.hdf')
    filter_terms = ['location == Global', 'draw == 10', 'fake_filter']

    a = Artifact(path, filter_terms)

    for key in _KEYS:
        a.load(key)
        assert a.filter_terms == filter_terms


def test_EntityKey_init_failure():
    bad_keys = ['hello', 'a.b.c.d', '', '.', '.coconut', 'a.', 'a..c']

    for k in bad_keys:
        with pytest.raises(ValueError):
            EntityKey(k)


def test_EntityKey_no_name():
    type_ = 'population'
    measure = 'structure'
    key = EntityKey(f'{type_}.{measure}')

    assert key.type == type_
    assert key.name == ''
    assert key.measure == measure
    assert key.group_prefix == '/'
    assert key.group_name == type_
    assert key.group == f'/{type_}'
    assert key.path == f'/{type_}/{measure}'
    assert key.with_measure('age_groups') == EntityKey('population.age_groups')


def test_EntityKey_with_name():
    type_ = 'cause'
    name = 'diarrheal_diseases'
    measure = 'incidence'
    key = EntityKey(f'{type_}.{name}.{measure}')

    assert key.type == type_
    assert key.name == name
    assert key.measure == measure
    assert key.group_prefix == f'/{type_}'
    assert key.group_name == name
    assert key.group == f'/{type_}/{name}'
    assert key.path == f'/{type_}/{name}/{measure}'
    assert key.with_measure('prevalence') == EntityKey(f'{type_}.{name}.prevalence')


def test_to_tree():
    keys = [EntityKey('population.structure'),
            EntityKey('population.age_groups'),
            EntityKey('cause.diarrheal_diseases.incidence'),
            EntityKey('cause.diarrheal_diseases.prevalence')]

    key_tree = {
        'population': {
            'structure': [],
            'age_groups': [],
        },
        'cause': {
            'diarrheal_diseases': ['incidence', 'prevalence']
        }
    }

    assert _to_tree(keys) == key_tree
