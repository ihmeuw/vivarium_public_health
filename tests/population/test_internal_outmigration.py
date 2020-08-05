from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from vivarium import InteractiveContext
from vivarium_public_health.population.spenser_population import TestPopulation, compute_migration_rates, transform_rate_table
from vivarium_public_health.population import IntegralOutMigration


@pytest.fixture()
def config(base_config):

    # change this to you own path
    path_dir= 'persistant_data/'

    # file should have columns -> PID,location,sex,age,ethnicity
    filename_pop = 'Testfile.csv'

    filename_internal_outmigration_name = 'InternalOutmig2011_LEEDS2.csv'
    path_to_internal_outmigration_file = "{}/{}".format(path_dir, filename_internal_outmigration_name)

    filename_total_population = 'MY2011AGEN.csv'
    path_to_total_population_file = "{}/{}".format(path_dir, filename_total_population)

    path_to_pop_file= "{}/{}".format(path_dir,filename_pop)

    pop_size = len(pd.read_csv(path_to_pop_file))

    base_config.update({

        'path_to_pop_file':path_to_pop_file,
        'path_to_internal_outmigration_file': path_to_internal_outmigration_file,
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



def test_internal_outmigration(config, base_plugins):
    start_population_size = config.population.population_size

    num_days = 365*5
    components = [TestPopulation(), IntegralOutMigration()]
    simulation = InteractiveContext(components=components,
                                    configuration=config,
                                    plugin_configuration=base_plugins,
                                    setup=False)


    # setup emigration rates

    df = pd.read_csv(config.path_to_internal_outmigration_file)

    # to save time, only look at locatiosn existing on the test dataset.
    df_internal_outmigration = df[(df['LAD.code'] == 'E09000002') | (df['LAD.code'] == 'E09000003')]

    asfr_data = transform_rate_table(df_internal_outmigration, 2011, 2012, config.population.age_start,
                                     config.population.age_end)

    simulation._data.write("cause.age_specific_internal_outmigration_rate", asfr_data)



    simulation.setup()


    simulation.run_for(duration=pd.Timedelta(days=num_days))
    pop = simulation.get_population()

    print ('internal outmigration',len(pop[pop['internal_outmigration']=='Yes']))
    print ('remaining population',len(pop[pop['internal_outmigration']=='No']))

    assert (np.all(pop.internal_outmigration == 'Yes') == False)

    assert len(pop[pop['last_outmigration_time']!='NaT']) > 0, 'time of out migration gets saved.'

