import os

from ceam import config

from ceam_tests.util import build_table, setup_simulation, generate_test_population, pump_simulation

from ceam_public_health.metrics.calculate_incidence import CalculateIncidence

def setup():
    # Remove user overrides but keep custom cache locations if any
    try:
        config.reset_layer('override', preserve_keys=['input_data.intermediary_data_cache_path',
                                                      'input_data.auxiliary_data_folder'])
    except KeyError:
        pass
    config.simulation_parameters.set_with_metadata('year_start', 2005, layer='override',
                                                   source=os.path.realpath(__file__))
    config.simulation_parameters.set_with_metadata('year_end', 2010, layer='override',
                                                   source=os.path.realpath(__file__))
    config.simulation_parameters.set_with_metadata('time_step', 365, layer='override',
                                                   source=os.path.realpath(__file__))
    config.simulation_parameters.set_with_metadata('num_simulants', 1000, layer='override',
                                                   source=os.path.realpath(__file__))
    return config
    
def test_calculate_incidence():
    simulation = setup_simulation([generate_test_population, CalculateIncidence(disease_col='diarrhea', disease='diarrhea', disease_states='mild_diarrhea')])

    pop = simulation.population.population

    pop['diarrhea'] = ['healthy'] * 500 + ['mild_diarrhea'] * 500

    pump_simulation(simulation, duration=timedelta(days=1))


    # get_value 'epidemiological_span_measures'
