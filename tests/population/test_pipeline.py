from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from vivarium import InteractiveContext
from vivarium_public_health.population.spenser_population import TestPopulation, build_mortality_table, transform_rate_table
from vivarium_public_health.population import Mortality
from vivarium_public_health.population import FertilityAgeSpecificRates


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


    path_to_pop_file= "{}/{}".format(path_dir,filename_pop)
    path_to_mortality_file= "{}/{}".format(path_dir,filename_mortality_rate)

    pop_size = len(pd.read_csv(path_to_pop_file))

    base_config.update({

        'path_to_pop_file':path_to_pop_file,
        'path_to_mortality_file': path_to_mortality_file,
        'path_to_fertility_file': path_to_fertility_file,

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

    num_days = 365*3
    components = [TestPopulation(), FertilityAgeSpecificRates() ,Mortality()]
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
    # Mock Fertility Data
    simulation._data.write("covariate.age_specific_fertility_rate.estimate", asfr_data_fertility)

    simulation.setup()
    time_start = simulation._clock.time


    assert 'last_birth_time' in simulation.get_population().columns, \
        'expect Fertility module to update state table.'
    assert 'parent_id' in simulation.get_population().columns, \
        'expect Fertility module to update state table.'

    simulation.run_for(duration=pd.Timedelta(days=num_days))
    pop = simulation.get_population()

    print ('alive',len(pop[pop['alive']=='alive']))
    print ('dead',len(pop[pop['alive']!='alive']))

    assert (np.all(pop.alive == 'alive') == False)

    assert len(pop.age) > start_population_size, 'expect new simulants'

    for i in range(start_population_size, len(pop)):
        assert pop.loc[pop.iloc[i].parent_id].last_birth_time >= time_start, 'expect all children to have mothers who' \
                                                                             ' gave birth after the simulation starts.'
