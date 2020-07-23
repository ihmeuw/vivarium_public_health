from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from vivarium import InteractiveContext
from vivarium_public_health.population.spenser_population import TestPopulation, compute_migration_rates
from vivarium_public_health.population import Emigration


@pytest.fixture()
def config(base_config):

    # change this to you own path
    path_dir= 'persistant_data/'

    # file should have columns -> PID,location,sex,age,ethnicity
    filename_pop = 'Testfile.csv'

    filename_emigration_name = 'Emig_2011_2012_LEEDS2.csv'
    path_to_emigration_file = "{}/{}".format(path_dir, filename_emigration_name)

    filename_total_population = 'MY2011AGEN.csv'
    path_to_total_population_file = "{}/{}".format(path_dir, filename_total_population)

    path_to_pop_file= "{}/{}".format(path_dir,filename_pop)

    pop_size = len(pd.read_csv(path_to_pop_file))

    base_config.update({

        'path_to_pop_file':path_to_pop_file,
        'path_to_emigration_file': path_to_emigration_file,
        'path_to_total_population_file': path_to_total_population_file,

        'population': {
            'population_size': pop_size,
            'age_start': 0,
            'age_end': 100,
        },
        'time': {
            'step_size': 10,
            },
        }, source=str(Path(__file__).resolve()))
    return base_config



def test_emigration(config, base_plugins):
    start_population_size = config.population.population_size

    num_days = 365*10
    components = [TestPopulation(), Emigration()]
    simulation = InteractiveContext(components=components,
                                    configuration=config,
                                    plugin_configuration=base_plugins,
                                    setup=False)


    # setup emigration rates

    df_emigration = pd.read_csv(config.path_to_emigration_file)
    df_total_population = pd.read_csv(config.path_to_total_population_file)
    df_emigration = df_emigration[
        (df_emigration['LAD.code'] == 'E09000002') | (df_emigration['LAD.code'] == 'E09000003')]
    df_total_population = df_total_population[
        (df_total_population['LAD'] == 'E09000002') | (df_total_population['LAD'] == 'E09000003')]
    asfr_data_emigration = compute_migration_rates(df_emigration, df_total_population, 2011, 2012, config.population.age_start, config.population.age_end,aggregate_over=75)
    # Mock emigration Data
    simulation._data.write("covariate.age_specific_migration_rate.estimate", asfr_data_emigration)

    simulation.setup()


    simulation.run_for(duration=pd.Timedelta(days=num_days))
    pop = simulation.get_population()

    print ('emigrated',len(pop[pop['alive']=='emigrated']))
    print ('remaining population',len(pop[pop['emigrated']=='no_emigrated']))

    assert (np.all(pop.alive == 'alive') == False)

    assert len(pop[pop['emigrated']=='Yes']) > 0, 'expect migration'

