from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from vivarium import InteractiveContext
from vivarium_public_health.population.spenser_population import TestPopulation, compute_migration_rates, transform_rate_table
from vivarium_public_health.population import Mortality
from vivarium_public_health.population import FertilityAgeSpecificRates
from vivarium_public_health.population import Emigration
from vivarium_public_health.population import ImmigrationDeterministic as Immigration



@pytest.fixture()
def config(base_config):

    # change this to you own path
    path_dir= 'persistant_data/'

    # file should have columns -> PID,location,sex,age,ethnicity
    filename_pop = 'Testfile.csv'
    # mortality file provided by N. Lomax
    filename_mortality_rate = 'Mortality2011_LEEDS1_2.csv'

    filename_fertility_rate = 'Fertility2011_LEEDS1_2.csv'
    path_to_fertility_file = "{}/{}".format(path_dir, filename_fertility_rate)

    filename_emigration_name = 'Emig_2011_2012_LEEDS2.csv'
    path_to_emigration_file = "{}/{}".format(path_dir, filename_emigration_name)

    filename_total_population = 'MY2011AGEN.csv'
    path_to_total_population_file = "{}/{}".format(path_dir, filename_total_population)

    # immigration file provided by N. Lomax
    filename_immigration_rate = 'Immig_2011_2012_LEEDS2.csv'
    path_to_immigration_file = "{}/{}".format(path_dir, filename_immigration_rate)

    path_to_pop_file= "{}/{}".format(path_dir,filename_pop)
    path_to_mortality_file= "{}/{}".format(path_dir,filename_mortality_rate)
    path_to_immigration_file = "{}/{}".format(path_dir, filename_immigration_rate)

    pop_size = len(pd.read_csv(path_to_pop_file))

    base_config.update({

        'path_to_pop_file':path_to_pop_file,
        'path_to_mortality_file': path_to_mortality_file,
        'path_to_fertility_file': path_to_fertility_file,
        'path_to_emigration_file': path_to_emigration_file,
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



def test_pipeline(config, base_plugins):
    start_population_size = config.population.population_size

    num_days = 365*2
    components = [TestPopulation(), FertilityAgeSpecificRates(), Mortality(), Emigration(), Immigration()]
    simulation = InteractiveContext(components=components,
                                    configuration=config,
                                    plugin_configuration=base_plugins,
                                    setup=False)


    # setup mortality rates
    df = pd.read_csv(config.path_to_mortality_file)
    mortality_rate_df = df[(df['LAD.code']=='E09000002') | (df['LAD.code']=='E09000003')]
    asfr_data = transform_rate_table(mortality_rate_df, 2011, 2012, config.population.age_start, config.population.age_end)
    simulation._data.write("cause.all_causes.cause_specific_mortality_rate", asfr_data)

    # setup fertility rates
    df_fertility = pd.read_csv(config.path_to_fertility_file)
    fertility_rate_df = df_fertility[(df_fertility['LAD.code'] == 'E09000002') | (df_fertility['LAD.code'] == 'E09000003')]
    asfr_data_fertility = transform_rate_table(fertility_rate_df, 2011, 2012, 10, 50, [2])
    simulation._data.write("covariate.age_specific_fertility_rate.estimate", asfr_data_fertility)

    # setup emigration rates
    df_emigration = pd.read_csv(config.path_to_emigration_file)
    df_total_population = pd.read_csv(config.path_to_total_population_file)
    df_emigration = df_emigration[
        (df_emigration['LAD.code'] == 'E09000002') | (df_emigration['LAD.code'] == 'E09000003')]
    df_total_population = df_total_population[
        (df_total_population['LAD'] == 'E09000002') | (df_total_population['LAD'] == 'E09000003')]
    asfr_data_emigration = compute_migration_rates(df_emigration, df_total_population, 2011, 2012, config.population.age_start, config.population.age_end)
    simulation._data.write("covariate.age_specific_migration_rate.estimate", asfr_data_emigration)

    # setup immigration rates
    df_immigration = pd.read_csv(config.path_to_immigration_file)
    df_immigration = df_immigration[
        (df_immigration['LAD.code'] == 'E09000002') | (df_immigration['LAD.code'] == 'E09000003') ]

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
    time_start = simulation._clock.time


    assert 'last_birth_time' in simulation.get_population().columns, \
        'expect Fertility module to update state table.'
    assert 'parent_id' in simulation.get_population().columns, \
        'expect Fertility module to update state table.'

    simulation.run_for(duration=pd.Timedelta(days=num_days))
    pop = simulation.get_population()

    print ('alive',len(pop[pop['alive']=='alive']))
    print ('dead',len(pop[pop['alive']=='dead']))
    print ('emigrated',len(pop[pop['alive']=='emigrated']))

    assert (np.all(pop.alive == 'alive') == False)
    assert len(pop[pop['emigrated']=='Yes']) > 0, 'expect migration'


    assert len(pop.age) > start_population_size, 'expect new simulants'

    for i in range(start_population_size, len(pop)):
        # skip immigrated population
        if pop.loc[i].immigrated == 'Yes':
            continue
        assert pop.loc[pop.loc[i].parent_id].last_birth_time >= time_start, 'expect all children to have mothers who' \
                                                                             ' gave birth after the simulation starts.'
