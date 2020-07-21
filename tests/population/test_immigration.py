from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from vivarium import InteractiveContext
from vivarium_public_health.population.spenser_population import TestPopulation
from vivarium_public_health.population.spenser_population import transform_rate_table
from vivarium_public_health.population.spenser_population import prepare_dataset
from vivarium_public_health.population.spenser_population import compute_migration_rates
from vivarium_public_health.population import ImmigrationDeterministic as Immigration



@pytest.fixture()
def config(base_config):

    # change this to you own path
    path_dir = 'persistant_data/'

    # file should have columns -> PID,location,sex,age,ethnicity
    filename_pop = 'Testfile.csv'
    # immigration file provided by N. Lomax
    filename_immigration_rate = 'Immig_2011_2012_LEEDS2.csv'
    filename_total_population = 'MY2011AGEN.csv'

    path_to_pop_file = "{}/{}".format(path_dir, filename_pop)
    path_to_immigration_file = "{}/{}".format(path_dir, filename_immigration_rate)
    path_to_total_population_file = "{}/{}".format(path_dir, filename_total_population)

    pop_size = len(pd.read_csv(path_to_pop_file))

    base_config.update({
        'path_to_pop_file': path_to_pop_file,
        'path_to_immigration_file': path_to_immigration_file,
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


def test_Immigration(config, base_plugins):
    num_days = 365
    components = [TestPopulation(), Immigration()]
    simulation = InteractiveContext(components=components,
                                    configuration=config,
                                    plugin_configuration=base_plugins,
                                    setup=False)

    df_total_population = pd.read_csv(config.path_to_total_population_file)
    df_total_population = df_total_population[
        (df_total_population['LAD'] == 'E09000002')]
    
    # setup immigration rates
    df_immigration = pd.read_csv(config.path_to_immigration_file)
    df_immigration = df_immigration[
        (df_immigration['LAD.code'] == 'E09000002')]
    
    asfr_data_immigration = compute_migration_rates(df_immigration, df_total_population, 
                                                    2011, 
                                                    2012, 
                                                    config.population.age_start, 
                                                    config.population.age_end,
                                                    normalize=False
                                                   )

    # read total immigrants from the file
    total_immigrants = int(df_immigration[df_immigration.columns[4:]].sum().sum())

    simulation._data.write("cause.all_causes.cause_specific_immigration_rate", asfr_data_immigration)
    simulation._data.write("cause.all_causes.cause_specific_total_immigrants_per_year", total_immigrants)

    simulation.setup()
    simulation.run_for(duration=pd.Timedelta(days=num_days))
    pop = simulation.get_population()

    assert (len(pop["entrance_time"].value_counts()) > 1)