import os

import numpy as np, pandas as pd
import pytest

from vivarium.test_util import setup_simulation, pump_simulation, assert_rate, build_table, TestPopulation

from ceam_public_health.dataset_manager import ArtifactManager
from ceam_public_health.testing.mock_artifact import MockArtifact
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
    base_config.run_configuration.set_with_metadata('input_draw_number', 1, **metadata)
    base_config.vivarium.set_with_metadata('dataset_manager', 'ceam_public_health.dataset_manager.ArtifactManager', **metadata)
    base_config.input_data.set_with_metadata('artifact_path', '/tmp/dummy.hdf', **metadata)
    base_config.artifact.set_with_metadata('artifact_class', 'ceam_public_health.testing.mock_artifact.MockArtifact', **metadata)
    return base_config


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
def test_general_access(config):
    year_start = config.time.start.year
    year_end = config.time.end.year

    artifact = MockArtifact()

    utilization_rate = build_table(0.1*12, year_start, year_end, ['age', 'year', 'sex', 'value'])
    artifact.set("healthcare_entity.outpatient_visits.annual_visits", utilization_rate)

    metrics = Metrics()
    simulation = setup_simulation([TestPopulation(), metrics, HealthcareAccess()], input_config=config, dataset_manager=ArtifactManager(artifact))

    # 1.2608717447575932 == a monthly probability 0.1 as a yearly rate
    assert_rate(simulation, 1.2608717447575932, lambda s: metrics.access_count)


@pytest.mark.slow
@pytest.mark.skip("I don't know why this is broken or how it works. -J.C.")
def test_adherence(config, get_annual_visits_mock):
    year_start = config.time.start.year
    year_end = config.time.end.year
    n_simulants = int('10_000')

    def get_utilization_rate(*_, **__):
        return build_table(0.1, year_start, year_end, ['age', 'year', 'sex', 'utilization_proportion'])
    get_annual_visits_mock.side_effect = get_utilization_rate

    metrics = Metrics()
    simulation = setup_simulation([TestPopulation(), metrics, HealthcareAccess()], input_config=config, population_size=n_simulants)

    t_step = 28 # days
    n_days = 28*2
    pump_simulation(simulation, time_step_days=t_step, duration=pd.Timedelta(days=n_days))

    df = simulation.population.population
    df['fu_visit'] = df.healthcare_visits > 1
    t = df.groupby('adherence_category').fu_visit.count()
    assert t['non-adherent'] == 0, 'non-adherents should not show for follow-up visit'
    assert t['semi-adherent'] < .9*t['adherent'], 'semi-adherents should show up less than adherents for follow-up visit'
