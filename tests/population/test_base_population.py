import os
from datetime import datetime

import numpy as np
import pandas as pd

from ceam import config
from ceam.framework.randomness import RandomnessStream

from ceam.test_util import setup_simulation, pump_simulation

from ceam_public_health.population.base_population import (generate_ceam_population, _add_proportions,
                                                           age_out_simulants, BasePopulation, age_simulants)
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
    pop = pop[pop.year == 1990]
    pop = generate_ceam_population(pop,
                                   number_of_simulants=1000000,
                                   randomness_stream=randomness)

    num_7_and_half_yr_old_males = pop.query("age == 7.5 and sex == 'Male'").copy()
    num_7_and_half_yr_old_males.loc[:, 'count'] = 1
    val = num_7_and_half_yr_old_males.groupby('age')[['count']].sum()
    val = val.get_value(7.5, 'count')
    val = val / 1000000

    assert np.isclose(val, 0.0823207075530383, atol=.01), ("there should be about 8.23% 7.5 year old males in "
                                                           "Kenya in 1990, based on data uploaded by em 1/5/2017")


def test_age_out_simulants():
    start_population_size = 1000
    num_days = 600
    time_step = 100  # Days
    time_start = pd.Timestamp('1990-01-01')
    config.read_dict({'simulation_parameters': {'initial_age': 4,
                                                'maximum_age': 5,
                                                'time_step': time_step}},
                     layer='override')
    components = [BasePopulation(), age_out_simulants, age_simulants]
    simulation = setup_simulation(components, population_size=start_population_size, start=time_start)
    pump_simulation(simulation, time_step_days=time_step, duration=pd.Timedelta(days=num_days))
    assert np.all(simulation.population.population.alive == 'untracked')



