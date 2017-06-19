import os
from datetime import timedelta

from ceam import config

from ceam_tests.util import build_table, setup_simulation, generate_test_population, pump_simulation

from ceam_public_health.metrics.calculate_incidence import CalculateIncidence
from ceam_public_health.metrics.epidemiology import EpidemiologicalMeasures
from ceam_public_health.experiments.diarrhea.components.diarrhea import diarrhea_factory

def setup():
    # Remove user overrides but keep custom cache locations if any
    try:
        config.reset_layer('override', preserve_keys=['input_data.intermediary_data_cache_path',
                                                      'input_data.auxiliary_data_folder'])
    except KeyError:
        pass
    config.simulation_parameters.set_with_metadata('year_start', 2009, layer='override',
                                                   source=os.path.realpath(__file__))
    config.simulation_parameters.set_with_metadata('year_end', 2011, layer='override',
                                                   source=os.path.realpath(__file__))
    config.simulation_parameters.set_with_metadata('time_step', 365, layer='override',
                                                   source=os.path.realpath(__file__))
    config.simulation_parameters.set_with_metadata('num_simulants', 1000, layer='override',
                                                   source=os.path.realpath(__file__))
    return config

# FIXME: test_calculate_incidence isn't testing anything right now. need to figure out how to access the incidence rate value in epidemiological_span_measures
@pytest.mark.xfail
def test_calculate_incidence():
    factory = diarrhea_factory()

    simulation = setup_simulation([generate_test_population, CalculateIncidence(disease_col='diarrhea', disease='diarrhea', disease_states=['mild_diarrhea']), EpidemiologicalMeasures()] + factory)

    simulation.population.population['diarrhea'] = ['healthy'] * 50 + ['mild_diarrhea'] * 50

    pump_simulation(simulation, duration=timedelta(days=730))

    inc = simulation.values.get_value('epidemiological_span_measures')

    # assert inc = .5
