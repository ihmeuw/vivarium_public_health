import os

import numpy as np, pandas as pd
import pytest

from vivarium.test_util import assert_rate, build_table, TestPopulation
from vivarium.interface.interactive import setup_simulation

from ceam_public_health.treatment import HealthcareAccess

np.random.seed(100)


@pytest.fixture(scope='function')
def config(base_config):
    try:
        base_config.reset_layer('override', preserve_keys=['input_data.intermediary_data_cache_path',
                                                           'input_data.auxiliary_data_folder'])
    except KeyError:
        pass
    metadata = {'layer': 'override', 'source': os.path.realpath(__file__)}
    base_config.time.start.set_with_metadata('year', 1995, **metadata)
    base_config.time.end.set_with_metadata('year', 2010, **metadata)
    base_config.time.set_with_metadata('step_size', 30.5, **metadata)
    base_config.input_data.set_with_metadata('input_draw_number', 1, **metadata)
    return base_config


@pytest.fixture(scope='function')
def get_annual_visits_mock(mocker):
    return mocker.patch('ceam_public_health.treatment.healthcare_access.get_healthcare_annual_visits')


class Metrics:
    def __init__(self):
        self.access_count = 0

    def setup(self, builder):
        builder.event.register_listener('general_healthcare_access', self.count_access)

    def count_access(self, event):
        self.access_count += len(event.index)

    def reset(self):
        self.access_count = 0


@pytest.mark.slow
def test_general_access(config, get_annual_visits_mock):
    year_start = config.time.start.year
    year_end = config.time.end.year

    def get_utilization_rate(*_, **__):
        return build_table(0.1*12, year_start, year_end, ['age', 'year', 'sex', 'annual_visits'])
    get_annual_visits_mock.side_effect = get_utilization_rate

    metrics = Metrics()
    simulation = setup_simulation([TestPopulation(), metrics, HealthcareAccess()], input_config=config)

    # 1.2608717447575932 == a monthly probability 0.1 as a yearly rate
    assert_rate(simulation, 1.2608717447575932, lambda s: metrics.access_count)


@pytest.mark.slow
@pytest.mark.skip("I don't know why this is broken or how it works. -J.C.")
def test_adherence(config, get_annual_visits_mock):
    year_start = config.time.start.year
    year_end = config.time.end.year
    t_step = 28  # days
    n_simulants = int('10_000')

    def get_utilization_rate(*_, **__):
        return build_table(0.1, year_start, year_end, ['age', 'year', 'sex', 'utilization_proportion'])
    get_annual_visits_mock.side_effect = get_utilization_rate

    metrics = Metrics()
    config.update({'population': {'population_size': n_simulants},
                   'time': {'step_size': t_step}}, layer='override')
    simulation = setup_simulation([TestPopulation(), metrics, HealthcareAccess()], input_config=config)

    simulation.take_steps(number_of_steps=2)

    df = simulation.population.population
    df['fu_visit'] = df.healthcare_visits > 1
    t = df.groupby('adherence_category').fu_visit.count()
    assert t['non-adherent'] == 0, 'non-adherents should not show for follow-up visit'
    assert t['semi-adherent'] < .9*t['adherent'], 'semi-adherents should show up less than adherents for follow-up visit'
