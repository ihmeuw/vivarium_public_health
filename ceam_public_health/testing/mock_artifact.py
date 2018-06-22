import pandas as pd

from vivarium.test_util import build_table

from ceam_public_health.dataset_manager import Artifact, ArtifactException
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
            'exposure': 0,
            'exposure_standard_deviation': 0.001,
            'relative_risk': build_table([1, "continuous"], 1990, 2018, ("age", "sex", "year", "value", "parameter")),
            'mediation_factor': pd.DataFrame({"value": [0]}),
            'tmred': lambda *args, **kwargs: {
                "distribution": "uniform",
                "min": 0,
                "max": 100,
                "inverted": False,
            },
            'exposure_parameters': lambda *args, **kwargs: {
                'scale': 1,
                'max_rr': 10,
                'max_val': 200,
                'min_val': 0,
            },
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
        super(MockArtifact, self).__init__("", None, None, 0, "")
        self._is_open = False
        self._overrides = {
                "risk_factor.correlations.correlations": pd.DataFrame([], columns=["risk_factor", "sex", "age"]),
        }

    def load(self, entity_path, keep_age_group_edges=False, **column_filters):
        if entity_path in self._overrides:
            value = self._overrides[entity_path]
        else:
            entity_type, *tail = entity_path.split('.')
            assert entity_type in MOCKERS
            assert tail[-1] in MOCKERS[entity_type]

            value = MOCKERS[entity_type][tail[-1]]
            if not callable(value) and not isinstance(value, (pd.DataFrame, pd.Series)):
                value = build_table(value, 1990, 2018)

        if callable(value):
            value = value(entity_path, keep_age_group_edges, **column_filters)

        return value

    def set(self, entity_path, value):
        self._overrides[entity_path] = value

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
