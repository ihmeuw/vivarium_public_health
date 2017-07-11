import os
from unittest.mock import patch

import numpy as np
import pytest

from vivarium import config
from vivarium.framework.event import listens_for
from vivarium.test_util import setup_simulation, assert_rate, build_table, generate_test_population

from ceam_public_health.population import adherence
from ceam_public_health.treatment import HealthcareAccess

np.random.seed(100)


def setup():
    try:
        config.reset_layer('override', preserve_keys=['input_data.intermediary_data_cache_path',
                                                      'input_data.auxiliary_data_folder'])
    except KeyError:
        pass
    config.simulation_parameters.set_with_metadata('year_start', 1990, layer='override',
                                                   source=os.path.realpath(__file__))
    config.simulation_parameters.set_with_metadata('year_end', 2010, layer='override',
                                                   source=os.path.realpath(__file__))
    config.simulation_parameters.set_with_metadata('time_step', 30.5, layer='override',
                                                   source=os.path.realpath(__file__))


class Metrics:
    def setup(self, builder):
        self.access_count = 0

    @listens_for('general_healthcare_access')
    def count_access(self, event):
        self.access_count += len(event.index)

    def reset(self):
        self.access_count = 0


@pytest.mark.slow
@patch('ceam_public_health.treatment.healthcare_access.get_proportion')
def test_general_access(utilization_rate_mock):
    utilization_rate_mock.side_effect = lambda *args, **kwargs: build_table(0.1, ['age', 'year', 'sex', 'utilization_proportion'])
    metrics = Metrics()
    simulation = setup_simulation([generate_test_population, adherence, metrics, HealthcareAccess()])

    # 1.2608717447575932 == a monthly probability 0.1 as a yearly rate
    assert_rate(simulation, 1.2608717447575932, lambda s: metrics.access_count)


#TODO: get fixture data for the cost table so we can test in a stable space
#@pytest.mark.slow
#def test_general_access_cost():
#    metrics = MetricsModule()
#    access = HealthcareAccessModule()
#    simulation = simulation_factory([metrics, access])
#
#    simulation.reset_population()
#    timestep = timedelta(days=30)
#    start_time = datetime(1990, 1, 1)
#    simulation.current_time = start_time
#
#    simulation._step(timestep)
#    simulation._step(timestep)
#    simulation._step(timestep)
#
#    assert np.allclose(sum(access.cost_by_year.values()) / metrics.access_count, access.appointment_cost[1990])


# End.
