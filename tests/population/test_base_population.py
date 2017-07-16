import os
from datetime import datetime
from unittest.mock import patch
import numpy as np
import pandas as pd

from vivarium import config
from vivarium.framework.randomness import RandomnessStream
from vivarium.test_util import setup_simulation, pump_simulation

from ceam_inputs import get_populations

from ceam_public_health.population.base_population import age_out_simulants, BasePopulation, generate_ceam_population
from ceam_public_health.population.data_transformations import assign_demographic_proportions


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


@patch('ceam_public_health.population.base_population.generate_ceam_population')
@patch('ceam_public_health.population.base_population._build_population_data_table')
def test_BasePopulation(build_pop_data_table_mock, generate_ceam_population_mock):
    start_population_size = 1000
    num_days = 600
    time_step = 100  # Days
    time_start = pd.Timestamp('1990-01-01')

    build_pop_data_table_mock.return_value = pd.DataFrame(
        {'age': [5, 15, 25]*4,
         'age_group_start': [0, 10, 20]*4,
         'age_group_end': [10, 20, 30]*4,
         'sex': ['Male']*3*2 + ['Female']*3*2,
         'location_id': [1]*3 + [2]*3 + [1]*3 + [2]*3,
         'year': [2000]*12,  # TODO : Finish building the mock from here
         'pop_scaled': [100]*12,
         'P(sex, location_id | age, year)': [1/4]*12,
         'P(sex, location_id, age | year)': [1/12]*12}
    )
    generate_ceam_population_mock.return_value = None  # Some dead, some alive

    components = [BasePopulation()]
    simulation = setup_simulation(components, population_size=start_population_size, start=time_start)
    # assert build_pop_table called with 180 and false  (Do some config setup)
    # assert generate_ceam_population called with correct arguments
    # assert simulation population exists and is correct

    pump_simulation(simulation, time_step_days=time_step, duration=pd.Timedelta(days=num_days))
    # tests for aging simulants


def test_age_out_simulants():
    start_population_size = 1000
    num_days = 600
    time_step = 100  # Days
    time_start = pd.Timestamp('1990-01-01')
    config.read_dict({'simulation_parameters': {'initial_age': 4,
                                                'maximum_age': 5,
                                                'time_step': time_step}},
                     layer='override')
    components = [BasePopulation(), age_out_simulants]
    simulation = setup_simulation(components, population_size=start_population_size, start=time_start)
    pump_simulation(simulation, time_step_days=time_step, duration=pd.Timedelta(days=num_days))
    assert np.all(simulation.population.population.alive == 'untracked')


# Split into test with initial age and test without, make sure appropriate functions get called with appropriate args.
def test_generate_ceam_population():
    randomness = RandomnessStream('population_generation_test', clock=lambda: datetime(1990, 1, 1), seed=12345)
    pop = assign_demographic_proportions(get_populations(KENYA))
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


def test__assign_demography_with_initial_age():
    pass


def test__assign_demography_with_age_bounds():
    pass

def test__build_population_data_table():
    pass

def test__get_population_data():
    pass

# TODO: Adapt these tests for updated base population component.
"""
@patch('ceam_public_health.population.base_population.get_populations')
@patch('ceam_public_health.population.base_population.get_subregions')
def test_assign_subregions_with_subregions(get_subregions_mock, get_populations_mock):
    get_subregions_mock.side_effect = lambda location_id: [10, 11, 12]
    test_populations = {
            10: build_table(20, ['age', 'year', 'sex', 'pop_scaled']),
            11: build_table(30, ['age', 'year', 'sex', 'pop_scaled']),
            12: build_table(50, ['age', 'year', 'sex', 'pop_scaled']),
    }
    get_populations_mock.side_effect = lambda location_id, year, sex: test_populations[location_id]
    r = RandomnessStream('assign_sub_region_test', clock=lambda: datetime(1990, 1, 1), seed=12345)
    locations = assign_subregions(pd.Index(range(100000)), location=180, year=2005, randomness=r)

    counts = locations.value_counts()
    counts = np.array([counts[lid] for lid in [10, 11, 12]])
    counts = counts / counts.sum()
    assert np.allclose(counts, [.2, .3, .5], rtol=0.01)


@patch('ceam_public_health.population.base_population.get_populations')
@patch('ceam_public_health.population.base_population.get_subregions')
def test_assign_subregions_without_subregions(get_subregions_mock, get_populations_mock):
    get_subregions_mock.side_effect = lambda location_id: []
    test_populations = {
            190: build_table(100, ['age', 'year', 'sex', 'pop_scaled']),
    }
    get_populations_mock.side_effect = lambda location_id, year, sex: test_populations[location_id]
    r = RandomnessStream('assign_sub_region_test', clock=lambda: datetime(1990, 1, 1), seed=12345)
    locations = assign_subregions(pd.Index(range(100000)), location=190, year=2005, randomness=r)

    assert np.all(locations == 190)
"""
