import os
from datetime import datetime

import numpy as np

from ceam import config
from ceam.framework.randomness import RandomnessStream

from ceam_public_health.population.base_population import generate_ceam_population, _add_proportions
from ceam_inputs import get_populations

KENYA = 180


def setup():
    try:
        config.reset_layer('override', preserve_keys=['input_data.intermediary_data_cache_path',
                                                      'input_data.auxiliary_data_folder'])
    except KeyError:
        pass
    config.simulation_parameters.set_with_metadata('pop_age_start', 0, layer='override',
                                                   source=os.path.realpath(__file__))
    config.simulation_parameters.set_with_metadata('pop_age_end', 110, layer='override',
                                                   source=os.path.realpath(__file__))


def test_generate_ceam_population():
    randomness = RandomnessStream('population_generation_test', clock=lambda: datetime(1990, 1, 1), seed=12345)
    pop = _add_proportions(get_populations(KENYA))
    pop = generate_ceam_population(pop,
                                   year=1990,
                                   number_of_simulants=1000000,
                                   randomness_stream=randomness)

    num_7_and_half_yr_old_males = pop.query("age == 7.5 and sex_id == 1").copy()
    num_7_and_half_yr_old_males['count'] = 1
    val = num_7_and_half_yr_old_males.groupby('age')[['count']].sum()
    val = val.get_value(7.5, 'count')
    val = val / 1000000

    assert np.isclose(val, 0.0823207075530383, atol=.01), ("there should be about 8.23% 7.5 year old males in "
                                                           "Kenya in 1990, based on data uploaded by em 1/5/2017")