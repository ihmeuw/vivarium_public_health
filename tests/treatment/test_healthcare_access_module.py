import os

import numpy as np
import pytest

from vivarium.framework.event import listens_for
from vivarium.test_util import setup_simulation, assert_rate, build_table, TestPopulation

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
    base_config.simulation_parameters.set_with_metadata('year_start', 1995, **metadata)
    base_config.simulation_parameters.set_with_metadata('year_end', 2010, **metadata)
    base_config.simulation_parameters.set_with_metadata('time_step', 30.5, **metadata)
    return base_config


@pytest.fixture(scope='function')
def get_annual_visits_mock(mocker):
    return mocker.patch('ceam_public_health.treatment.healthcare_access.get_healthcare_annual_visits')


class Metrics:
    def __init__(self):
        self.access_count = 0

    @listens_for('general_healthcare_access')
    def count_access(self, event):
        self.access_count += len(event.index)

    def reset(self):
        self.access_count = 0


@pytest.mark.slow
def test_general_access(config, get_annual_visits_mock):
    year_start = config.simulation_parameters.year_start
    year_end = config.simulation_parameters.year_end

    def get_utilization_rate(*_, **__):
        return build_table(0.1*12, year_start, year_end, ['age', 'year', 'sex', 'annual_visits'])
    get_annual_visits_mock.side_effect = get_utilization_rate

    metrics = Metrics()
    simulation = setup_simulation([TestPopulation(), metrics, HealthcareAccess()], input_config=config)

    # 1.2608717447575932 == a monthly probability 0.1 as a yearly rate
    assert_rate(simulation, 1.2608717447575932, lambda s: metrics.access_count)
