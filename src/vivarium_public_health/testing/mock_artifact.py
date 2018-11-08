import pandas as pd

from vivarium.testing_utilities import build_table

from vivarium_public_health.dataset_manager import ArtifactManager, EntityKey
from .utils import make_uniform_pop_data

MOCKERS = {
        'cause': {
            'prevalence': 0,
            'cause_specific_mortality': 0,
            'excess_mortality': 0,
            'remission': 0,
            'incidence': 0.001,
            'disability_weight': pd.DataFrame({'value': [0]}),
        },
        'risk_factor': {
            'distribution': lambda *args, **kwargs: 'ensemble',
            'exposure': 120,
            'exposure_standard_deviation': 15,
            'relative_risk': build_table([1.5, "continuous", "test_cause"], 1990, 2018,
                                         ("age", "sex", "year", "value", "parameter", "cause")),
            'population_attributable_fraction': build_table([1, "test_cause_1"], 1990, 2018,
                                                            ("age", "sex", "year", "value", "cause")),
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
            'cost': build_table(0, 1990, 2018).query('sex=="Both" and age_group_start==27').drop('sex', 'columns'),
            'annual_visits': 0,
        },
        'population': {
            'structure': make_uniform_pop_data(),
            'theoretical_minimum_risk_life_expectancy': build_table(98.0, 1990, 1990).query('sex=="Both"')\
                .filter(['age_group_start', 'age_group_end', 'value'])
        },
}


class MockArtifact():

    def __init__(self):
        self.mocks = MOCKERS.copy()

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

    def write(self, entity_key, data):
        self.mocks[entity_key] = data


class MockArtifactManager(ArtifactManager):

    def __init__(self):
        self.artifact = self._load_artifact(None)

    def setup(self, builder):
        pass

    def load(self, entity_key, *args, **kwargs):
        return self.artifact.load(EntityKey(entity_key))

    def write(self, entity_key, data):
        self.artifact.write(EntityKey(entity_key), data)

    def _load_artifact(self, _):
        return MockArtifact()


