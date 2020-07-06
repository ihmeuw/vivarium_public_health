from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from vivarium import InteractiveContext
from vivarium_public_health.population.spenser_population import TestPopulation, build_mortality_table, transform_rate_table
from vivarium_public_health.population.spenser_population import prepare_dataset
from vivarium_public_health.population import Mortality


@pytest.fixture()
def config(base_config):

    # change this to you own path
    path_dir= 'persistant_data/'

    # read a dataset from daedalus, change columns to be readable by vivarium
    prepare_dataset(dataset_path="./persistant_data/1000rows_ssm_E08000032_MSOA11_ppp_2011.csv", 
                    output_path="./persistant_data/test_ssm_E08000032_MSOA11_ppp_2011.csv"
                   )

    # file should have columns -> PID,location,sex,age,ethnicity
    filename_pop = 'test_ssm_E08000032_MSOA11_ppp_2011.csv'
    # mortality file provided by N. Lomax
    filename_mortality_rate = 'Mortality2011_LEEDS1_2.csv'

    path_to_pop_file= "{}/{}".format(path_dir,filename_pop)
    path_to_mortality_file= "{}/{}".format(path_dir,filename_mortality_rate)

    pop_size = len(pd.read_csv(path_to_pop_file))

    base_config.update({

        'path_to_pop_file':path_to_pop_file,
        'path_to_mortality_file': path_to_mortality_file,

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



def test_Mortality(config, base_plugins):
    num_days = 365
    components = [TestPopulation(), Mortality()]
    simulation = InteractiveContext(components=components,
                                    configuration=config,
                                    plugin_configuration=base_plugins,
                                    setup=False)



    df = pd.read_csv(config.path_to_mortality_file)

    # to save time, only look at locatiosn existing on the test dataset.
    mortality_rate_df = df[df['LAD.code'] == 'E08000032']

    asfr_data = transform_rate_table(mortality_rate_df, 2011, 2012, config.population.age_start, config.population.age_end)

    simulation._data.write("cause.all_causes.cause_specific_mortality_rate", asfr_data)

    simulation.setup()
    simulation.run_for(duration=pd.Timedelta(days=num_days))
    pop = simulation.get_population()

    print ('alive',len(pop[pop['alive']=='alive']))
    print ('dead',len(pop[pop['alive']!='alive']))

    assert (np.all(pop.alive == 'alive') == False)

