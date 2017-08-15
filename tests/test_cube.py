from unittest.mock import patch

import numpy as np

from ceam_inputs import causes, sequelae
from vivarium.test_util import build_table
from ceam_public_health.cube import make_measure_cube_from_gbd


@patch('ceam_public_health.cube.get_prevalence')
@patch('ceam_public_health.cube.get_incidence')
@patch('ceam_public_health.cube.get_cause_specific_mortality')
def test_make_measure_cube(csmr_mock, incidence_mock, prevalence_mock):
    prevalence_dummies = {
            sequelae.heart_attack: build_table(0.5),
            sequelae.angina.severity_splits.mild: build_table(0.1),
    }
    prevalence_mock.side_effect = prevalence_dummies.get
    incidence_dummies = {
            sequelae.angina: build_table(0.4),
            causes.hemorrhagic_stroke: build_table(0.2),
    }
    incidence_mock.side_effect = incidence_dummies.get
    mortality_dummies = {
            causes.all_causes: build_table(0.6),
            causes.ischemic_heart_disease: build_table(0.3),
            causes.diarrhea: build_table(0.4),
    }
    csmr_mock.side_effect = mortality_dummies.get


    cube = make_measure_cube_from_gbd(1990, 2010, [180], [0], [
                                    ('heart_attack', 'prevalence'),
                                    ('mild_angina', 'prevalence'),
                                    ('angina', 'incidence'),
                                    ('hemorrhagic_stroke', 'incidence'),
                                    ('ischemic_heart_disease', 'csmr'),
                                    ('diarrhea', 'csmr'),
                                    ('all_causes', 'csmr'),
        ])

    cube = cube.reset_index()
    assert np.all(cube.query('cause == "heart_attack" and measure == "prevalence"').value == 0.5)
    assert np.all(cube.query('cause == "mild_angina" and measure == "prevalence"').value == 0.1)
    assert np.all(cube.query('cause == "angina" and measure == "incidence"').value == 0.4)
    assert np.all(cube.query('cause == "hemorrhagic_stroke" and measure == "incidence"').value == 0.2)
    assert np.all(cube.query('cause == "ischemic_heart_disease" and measure == "csmr"').value == 0.3)
    assert np.all(cube.query('cause == "diarrhea" and measure == "csmr"').value == 0.4)
    assert np.all(cube.query('cause == "all_causes" and measure == "csmr"').value == 0.6)
