from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from vivarium import InteractiveContext
from vivarium_public_health.population.spenser_population import TestPopulation, build_fertility_table, transform_rate_table

from vivarium_public_health import utilities
from vivarium_public_health.population import FertilityAgeSpecificRates


@pytest.fixture()
def config(base_config):
    # change this to you own path
    # IDEA make sure test files are contained within tests/data?
    path_dir = 'persistant_data'
    # file should have columns -> PID,location,sex,age,ethnicity
    filename = 'Testfile.csv'

    path_to_pop_file = "{}/{}".format(path_dir, filename)
    pop_size = len(pd.read_csv(path_to_pop_file))

    # mortality file provided by N. Lomax
    filename_fertility_rate = 'Fertility2011_LEEDS1_2.csv'
    path_to_fertility_file = "{}/{}".format(path_dir, filename_fertility_rate)


    # FIXME could push this config up to base config in conftest.py? Exists in mortality as well.
    base_config.update({
        'path_to_pop_file': path_to_pop_file,
        'path_to_fertility_file': path_to_fertility_file,

        'population': {
            'population_size': pop_size,
            'age_start': 0,
            'age_end': 100,
        },
        'time': {
            'step_size': 1,
        }
    }, source=str(Path(__file__).resolve()))
    return base_config


def test_fertility_module(config, base_plugins):
    start_population_size = config.population.population_size
    num_days = 365*3
    components = [TestPopulation(), FertilityAgeSpecificRates()]
    simulation = InteractiveContext(components=components,
                                    configuration=config,
                                    plugin_configuration=base_plugins,
                                    setup=False)

    df = pd.read_csv(config.path_to_fertility_file)

    # to save time, only look at locatiosn existing on the test dataset.
    fertility_rate_df = df[(df['LAD.code'] == 'E09000002') | (df['LAD.code'] == 'E09000003')]

    asfr_data = transform_rate_table(fertility_rate_df, 2011, 2012, 10,50,[2])

    # Mock Fertility Data
    simulation._data.write("covariate.age_specific_fertility_rate.estimate", asfr_data)

    simulation.setup()

    time_start = simulation._clock.time

    assert 'last_birth_time' in simulation.get_population().columns, \
        'expect Fertility module to update state table.'
    assert 'parent_id' in simulation.get_population().columns, \
        'expect Fertility module to update state table.'

    simulation.run_for(duration=pd.Timedelta(days=num_days))

    pop = simulation.get_population()
    # print(pop)

    # No death in this model.
    assert np.all(pop.alive == 'alive'), 'expect all simulants to be alive'

    # TODO: Write a more rigorous test.
    assert len(pop.age) > start_population_size, 'expect new simulants'

    for i in range(start_population_size, len(pop)):
        assert pop.loc[pop.iloc[i].parent_id].last_birth_time >= time_start, 'expect all children to have mothers who' \
                                                                             ' gave birth after the simulation starts.'
