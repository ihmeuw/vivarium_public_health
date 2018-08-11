import pandas as pd

from vivarium.testing_utilities import build_table

from vivarium_public_health.dataset_manager import Artifact, ArtifactException, ArtifactManager, EntityKey
from .utils import make_uniform_pop_data

MOCKERS = {
        'cause': {
            'prevalence': 0,
            'cause_specific_mortality': 0,
            'population_attributable_fraction': 1,
            'excess_mortality': 0,
            'remission': 0,
            'incidence': 0.001,
            'disability_weight': pd.DataFrame({'value': [0]}),
        },
        'risk_factor': {
            'distribution': lambda *args, **kwargs: 'ensemble',
            'exposure': 120,
            'exposure_standard_deviation': 15,
            'relative_risk': build_table([1.5, "continuous"], 1990, 2018, ("age", "sex", "year", "value", "parameter")),
            'tmred': lambda *args, **kwargs: {
                "distribution": "uniform",
                "min": 80,
                "max": 100,
                "inverted": False,
            },
            'exposure_parameters': lambda *args, **kwargs: {
                'scale': 1,
                'max_rr': 10,
                'max_val': 200,
                'min_val': 0,
            },
            'ensemble_weights': lambda *args, **kwargs: pd.DataFrame({'norm': 1}, index=[0])
        },
        'sequela': {
            'prevalence': 0,
            'cause_specific_mortality': 0,
            'excess_mortality': 0,
            'remission': 0,
            'incidence': 0.001,
            'disability_weight': pd.DataFrame({'value': [0]}),
        },
        'etiology': {
            'population_attributable_fraction': 1,
        },
        'healthcare_entity': {
            'cost': build_table(0, 1990, 2018).query('sex=="Both" and age==27').drop('sex', 'columns'),
            'annual_visits': 0,
        },
        'population': {
            'structure': make_uniform_pop_data(),
            'theoretical_minimum_risk_life_expectancy': 98,
        },
}


class MockArtifact(Artifact):
    def __init__(self):
        super().__init__("")
        self._is_open = False
        self.mocks = MOCKERS

    def load(self, entity_key):
        if entity_key in self.mocks:
            value = self.mocks[entity_key]
        else:
            assert entity_key.type in self.mocks
            assert entity_key.measure in self.mocks[entity_key.type]
            value = self.mocks[entity_key.type][entity_key.measure]

            if callable(value):
                value = value(entity_key)
            elif not isinstance(value, (pd.DataFrame, pd.Series)):
                value = build_table(value, 1990, 2018)

        return value

    def set(self, key, value):
        self.mocks[key] = value

    def open(self):
        if not self._is_open:
            self._is_open = True
        else:
            raise ArtifactException("Opening already open artifact")

    def close(self):
        if self._is_open:
            self._is_open = False
        else:
            raise ArtifactException("Closing already closed artifact")

    def summary(self):
        return "Mock Artifact"


class MockArtifactManager(ArtifactManager):

    def __init__(self):
        self.artifact = self._load_artifact(None, None)

    def setup(self, builder):
        self.artifact.open()
        builder.event.register_listener('post_setup', lambda _: self.artifact.close())

    def load(self, entity_key, *args, **kwargs):
        return self.artifact.load(EntityKey(entity_key))

    def set(self, key, value):
        self.artifact.set(key, value)

    def _load_artifact(self, artifact_path, base_filter_terms):
        return MockArtifact()


