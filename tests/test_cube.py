import pytest
import numpy as np

from ceam_inputs import causes, sequelae
from vivarium.test_util import build_table
from ceam_public_health.cube import make_measure_cube_from_gbd


@pytest.fixture(scope='function')
def csmr_mock(mocker):
    return mocker.patch('ceam_public_health.cube.get_cause_specific_mortality')


@pytest.fixture(scope='function')
def incidence_mock(mocker):
    return mocker.patch('ceam_public_health.cube.get_incidence')


@pytest.fixture(scope='function')
def prevalence_mock(mocker):
    return mocker.patch('ceam_public_health.cube.get_prevalence')


def test_make_measure_cube(base_config, csmr_mock, incidence_mock, prevalence_mock):
    year_start = base_config.time.start.year
    year_end = base_config.time.end.year

    prevalence_dummies = {
            causes.ischemic_heart_disease: build_table(0.5, year_start, year_end),
            sequelae.moderate_diarrheal_diseases: build_table(0.1, year_start, year_end),
    }
    prevalence_mock.side_effect = prevalence_dummies.get
    incidence_dummies = {
            sequelae.severe_angina_due_to_ischemic_heart_disease: build_table(0.4, year_start, year_end),
            causes.hemorrhagic_stroke: build_table(0.2, year_start, year_end),
    }
    incidence_mock.side_effect = incidence_dummies.get
    mortality_dummies = {
            causes.all_causes: build_table(0.6, year_start, year_end),
            causes.ischemic_heart_disease: build_table(0.3, year_start, year_end),
            causes.diarrheal_diseases: build_table(0.4, year_start, year_end),
    }
    csmr_mock.side_effect = mortality_dummies.get

    cube = make_measure_cube_from_gbd(1990, 2010, [180], [0], [
                                    (causes.ischemic_heart_disease.name, 'prevalence'),
                                    (sequelae.moderate_diarrheal_diseases.name, 'prevalence'),
                                    (sequelae.severe_angina_due_to_ischemic_heart_disease.name, 'incidence'),
                                    (causes.hemorrhagic_stroke.name, 'incidence'),
                                    (causes.all_causes.name, 'csmr'),
                                    (causes.diarrheal_diseases.name, 'csmr'),
                                    (causes.ischemic_heart_disease.name, 'csmr'),
        ], base_config)

    cube = cube.reset_index()
    assert np.all(cube.query('cause == @causes.ischemic_heart_disease.name '
                             'and measure == "prevalence"').value == 0.5)
    assert np.all(cube.query('cause == @sequelae.moderate_diarrheal_diseases.name '
                             'and measure == "prevalence"').value == 0.1)

    assert np.all(cube.query('cause == @sequelae.severe_angina_due_to_ischemic_heart_disease.name '
                             'and measure == "incidence"').value == 0.4)
    assert np.all(cube.query('cause == @causes.hemorrhagic_stroke.name '
                             'and measure == "incidence"').value == 0.2)

    assert np.all(cube.query('cause == @causes.all_causes.name '
                             'and measure == "csmr"').value == 0.6)
    assert np.all(cube.query('cause == @causes.diarrheal_diseases.name'
                             ' and measure == "csmr"').value == 0.4)
    assert np.all(cube.query('cause == @causes.ischemic_heart_disease.name'
                             ' and measure == "csmr"').value == 0.3)
