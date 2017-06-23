from unittest.mock import patch

import numpy as np

from ceam_inputs import causes
from ceam_tests.util import build_table
from ceam_public_health.cube import make_measure_cube_from_gbd


@patch('ceam_public_health.cube.get_prevalence')
@patch('ceam_public_health.cube.get_incidence')
@patch('ceam_public_health.cube.get_cause_specific_mortality')
@patch('ceam_public_health.cube.get_cause_deleted_mortality_rate')
def test_make_measure_cube(all_cause, csmr_mock, incidence_mock, prevalence_mock):
    prevalence_dummies = {
            causes.heart_attack.prevalence: build_table(0.5),
            causes.mild_angina.prevalence: build_table(0.1),
    }
    prevalence_mock.side_effect = prevalence_dummies.get
    incidence_dummies = {
            causes.angina_not_due_to_MI.incidence: build_table(0.4),
            causes.hemorrhagic_stroke.incidence: build_table(0.2),
    }
    incidence_mock.side_effect = incidence_dummies.get
    mortality_dummies = {
            causes.mild_heart_failure.mortality: build_table(0.3),
            causes.diarrhea.mortality: build_table(0.4),
    }
    csmr_mock.side_effect = mortality_dummies.get
    all_cause.side_effect = lambda x: build_table(0.6)

    cube = make_measure_cube_from_gbd(1990, 2010, [180], [0], [
                                    ('heart_attack', 'prevalence'),
                                    ('mild_angina', 'prevalence'),
                                    ('angina_not_due_to_MI', 'incidence'),
                                    ('hemorrhagic_stroke', 'incidence'),
                                    ('mild_heart_failure', 'mortality'),
                                    ('diarrhea', 'mortality'),
                                    ('all', 'mortality'),
        ])

    cube = cube.reset_index()
    assert np.all(cube.query('cause == "heart_attack" and measure == "prevalence"').value == 0.5)
    assert np.all(cube.query('cause == "mild_angina" and measure == "prevalence"').value == 0.1)
    assert np.all(cube.query('cause == "angina_not_due_to_MI" and measure == "incidence"').value == 0.4)
    assert np.all(cube.query('cause == "hemorrhagic_stroke" and measure == "incidence"').value == 0.2)
    assert np.all(cube.query('cause == "mild_heart_failure" and measure == "mortality"').value == 0.3)
    assert np.all(cube.query('cause == "diarrhea" and measure == "mortality"').value == 0.4)
    assert np.all(cube.query('cause == "all" and measure == "mortality"').value == 0.6)
