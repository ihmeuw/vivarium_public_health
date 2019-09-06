"""
==================
Mock Data Artifact
==================

This module contains a mock version of the artifact manager for use with
testing vivarium_public_health components.

"""
import pandas as pd

from vivarium.testing_utilities import build_table

from vivarium.framework.artifact import ArtifactManager

from .utils import make_uniform_pop_data

MOCKERS = {
        'cause': {
            'prevalence': 0,
            'cause_specific_mortality_rate': 0,
            'excess_mortality_rate': 0,
            'remission_rate': 0,
            'incidence_rate': 0.001,
            'disability_weight': pd.DataFrame({'value': [0]}),
            'restrictions': lambda *args, **kwargs: {'yld_only': False}
        },
        'risk_factor': {
            'distribution': lambda *args, **kwargs: 'ensemble',
            'exposure': 120,
            'exposure_standard_deviation': 15,
            'relative_risk': build_table([1.5, "continuous", "test_cause", "incidence_rate"], 1990, 2017,
                                         ("age", "sex", "year", "value", "parameter", "cause", "affected_measure")),
            'population_attributable_fraction': build_table([1, "test_cause_1", "incidence_rate"], 1990, 2017,
                                                            ("age", "sex", "year", "value", "cause", "affected_measure")),
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
            'cause_specific_mortality_rate': 0,
            'excess_mortality_rate': 0,
            'remission_rate': 0,
            'incidence_rate': 0.001,
            'disability_weight': pd.DataFrame({'value': [0]}),
        },
        'etiology': {
            'population_attributable_fraction': build_table([1, "incidence_rate"], 1990, 2017,
                                                            ("age", "sex", "year", "value", "affected_measure")),
        },
        'healthcare_entity': {
            'cost': build_table([0, 'outpatient_visits'], 1990, 2017,
                                ("age", "sex", "year", "value", "healthcare_entity")),
            'utilization_rate': 0,
        },
        'population': {
            'structure': make_uniform_pop_data(),
            'theoretical_minimum_risk_life_expectancy': (build_table(98.0, 1990, 1990)
                                                         .query('sex=="Female"')
                                                         .filter(['age_start', 'age_end', 'value']))
        },
}


class MockArtifact():

    def __init__(self):
        self.mocks = MOCKERS.copy()

    def load(self, entity_key):
        if entity_key in self.mocks:
            return self.mocks[entity_key]

        entity_type, *_, entity_measure = entity_key.split('.')
        assert entity_type in self.mocks
        assert entity_measure in self.mocks[entity_type]
        value = self.mocks[entity_type][entity_measure]

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

    @property
    def name(self):
        return 'mock_artifact_manager'

    def setup(self, builder):
        pass

    def load(self, entity_key, *args, **kwargs):
        return self.artifact.load(entity_key)

    def write(self, entity_key, data):
        self.artifact.write(entity_key, data)

    def _load_artifact(self, _):
        return MockArtifact()
