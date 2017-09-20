import os

import pytest
import pandas as pd

from vivarium.test_util import setup_simulation, TestPopulation, pump_simulation

from ceam_public_health.metrics.calculate_incidence import CalculateIncidence
from ceam_public_health.metrics.epidemiology import EpidemiologicalMeasures


@pytest.fixture(scope='function')
def config(base_config):
    try:
        base_config.reset_layer('override', preserve_keys=['input_data.intermediary_data_cache_path',
                                                           'input_data.auxiliary_data_folder'])
    except KeyError:
        pass

    metadata = {'layer': 'override', 'source': os.path.realpath(__file__)}
    base_config.simulation_parameters.set_with_metadata('year_start', 2009, **metadata)
    base_config.simulation_parameters.set_with_metadata('year_end', 2011, **metadata)
    base_config.simulation_parameters.set_with_metadata('time_step', 365, **metadata)
    base_config.population.set_with_metadata('population_size', 1000, **metadata)
    return base_config


# FIXME: test_calculate_incidence isn't testing anything right now. need to
# figure out how to access the incidence rate value in epidemiological_span_measures
@pytest.mark.skip
def test_calculate_incidence(config):
    factory = diarrhea_factory()
    ci = CalculateIncidence(disease_col='diarrhea', disease='diarrhea', disease_states=['mild_diarrhea'])
    simulation = setup_simulation([TestPopulation(), ci, EpidemiologicalMeasures()] + factory, input_config=config)
    simulation.population.population['diarrhea'] = ['healthy'] * 50 + ['mild_diarrhea'] * 50
    pump_simulation(simulation, duration=pd.Timedelta(days=730))

    inc = simulation.values.get_value('epidemiological_span_measures')

    # assert inc = .5
