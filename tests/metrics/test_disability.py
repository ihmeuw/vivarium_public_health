from collections import namedtuple
import os

import pytest
import numpy as np
import pandas as pd

from vivarium.test_util import setup_simulation, pump_simulation, build_table, TestPopulation

from ceam_public_health.population import Mortality
from ceam_public_health.disease import ExcessMortalityState, DiseaseModel, DiseaseState
from ceam_public_health.metrics import Metrics, Disability

Disease = namedtuple('Disease', 'name')

@pytest.fixture(scope='function')
def config(base_config):
    try:
        base_config.reset_layer('override', preserve_keys=['input_data.intermediary_data_cache_path',
                                                           'input_data.auxiliary_data_folder'])
    except KeyError:
        pass

    metadata = {'layer': 'override', 'source': os.path.realpath(__file__)}
    base_config.simulation_parameters.set_with_metadata('year_start', 1995, **metadata)
    base_config.simulation_parameters.set_with_metadata('year_end', 2000, **metadata)
    base_config.simulation_parameters.set_with_metadata('time_step', 30.5, **metadata)
    return base_config


def set_up_test_parameters(config, flu=False, mumps=False, deadly=False):
    """
    Sets up a simulation with specified disease states

    flu: bool
        If true, include an excess mortality state for flu
        If false, do not include an excess mortality state for flu

    mumps: bool
        If true, include an excess mortality state for mumps
        If false, do not include an excess mortality state for mumps
    """
    year_start = config.simulation_parameters.year_start
    year_end = config.simulation_parameters.year_end
    n_simulants = 1000

    asymp_data_funcs = {'prevalence': lambda _, __: build_table(1.0, year_start, year_end,
                                                                ['age', 'year', 'sex', 'prevalence']),
                        'disability_weight': lambda _, __: 0.0,
                        'dwell_time': lambda _, __: pd.Timedelta(days=0),
                        'excess_mortality': lambda _, __: build_table(0, year_start, year_end)}

    asymptomatic_disease_state = ExcessMortalityState('asymptomatic', get_data_functions=asymp_data_funcs)
    asymptomatic_disease_model = DiseaseModel(Disease(name='asymptomatic'),
                                              states=[asymptomatic_disease_state],
                                              initial_state='asymptomatic',
                                              get_data_functions={
                                                  'csmr': lambda _, __: build_table(0, year_start, year_end)})
    metrics = Metrics()
    disability = Disability()
    components = [TestPopulation(), asymptomatic_disease_model, metrics, disability]

    if flu:
        flu_data_funcs = {'prevalence': lambda _, __: build_table(1.0, year_start, year_end,
                                                                  ['age', 'year', 'sex', 'prevalence']),
                          'disability_weight': lambda _, __: 0.2,
                          'dwell_time': lambda _, __: pd.Timedelta(days=0),
                          'excess_mortality': lambda _, __: build_table(0, year_start, year_end)}
        flu = ExcessMortalityState('flu', get_data_functions=flu_data_funcs)
        flu_model = DiseaseModel(Disease(name='flu'), states=[flu],
                                 initial_state='flu',
                                 get_data_functions={'csmr': lambda _, __: build_table(0, year_start, year_end)})
        components.append(flu_model)

    if mumps:
        mumps_data_funcs = {'prevalence': lambda _, __: build_table(1.0, year_start, year_end,
                                                                    ['age', 'year', 'sex', 'prevalence']),
                            'disability_weight': lambda _, __: 0.4,
                            'dwell_time': lambda _, __: pd.Timedelta(days=0),
                            'excess_mortality': lambda _, __: build_table(0, year_start, year_end)}
        mumps = ExcessMortalityState('mumps', get_data_functions=mumps_data_funcs)
        mumps_model = DiseaseModel(Disease(name='mumps'), states=[mumps],
                                   initial_state='mumps',
                                   get_data_functions={'csmr': lambda _, __: build_table(0, year_start, year_end)})
        components.append(mumps_model)

    if deadly:
        deadly_data_funcs = {'prevalence': lambda _, __: build_table(0.1, year_start, year_end,
                                                                     ['age', 'year', 'sex', 'prevalence']),
                             'disability_weight': lambda _, __: 0.4,
                             'dwell_time': lambda _, __: pd.Timedelta(days=0),
                             'excess_mortality': lambda _, __: build_table(0.005, year_start, year_end)}
        deadly = ExcessMortalityState('deadly', get_data_functions=deadly_data_funcs)
        healthy = DiseaseState('healthy', get_data_functions=deadly_data_funcs)
        deadly_model = DiseaseModel(Disease(name='deadly'), states=[deadly, healthy], get_data_functions={
            'csmr': lambda _, __: build_table(0.0005, year_start, year_end)})
        components.append(deadly_model)
        components.append(Mortality())

    simulation = setup_simulation(components=components, population_size=n_simulants, input_config=config)

    return simulation, metrics, disability


def test_that_ylds_are_0_at_sim_beginning(config):
    simulation, metrics, disability = set_up_test_parameters(config)
    ylds = metrics.metrics(simulation.population.population.index)['years_lived_with_disability']
    assert ylds == 0


def test_that_healthy_people_dont_accrue_disability_weights(config):
    simulation, metrics, disability = set_up_test_parameters(config)
    pump_simulation(simulation, duration=pd.Timedelta(days=365))
    pop_size = len(simulation.population.population)
    ylds = metrics.metrics(simulation.population.population.index)['years_lived_with_disability']
    assert np.isclose(ylds, pop_size * 0.0, rtol=0.01)


def test_single_disability_weight(config):
    simulation, metrics, disability = set_up_test_parameters(config, flu=True)
    flu_dw = 0.2
    pump_simulation(simulation, duration=pd.Timedelta(days=365))
    pop_size = len(simulation.population.population)
    ylds = metrics.metrics(simulation.population.population.index)['years_lived_with_disability']
    assert np.isclose(ylds, pop_size * flu_dw, rtol=0.01)


def test_joint_disability_weight(config):
    simulation, metrics, disability = set_up_test_parameters(config, flu=True, mumps=True)
    flu_dw = 0.2
    mumps_dw = 0.4
    pump_simulation(simulation, duration=pd.Timedelta(days=365))
    pop_size = len(simulation.population.population)
    ylds = metrics.metrics(simulation.population.population.index)['years_lived_with_disability']
    # check that JOINT disability weight is correctly calculated
    assert np.isclose(ylds, pop_size * (1-(1-flu_dw)*(1-mumps_dw)), rtol=0.01)


@pytest.mark.skip(reason="Error in way csmr is being computed when using dataframes with inconsistent ages.")
def test_dead_people_dont_accrue_disability(config):
    simulation, metrics, disability = set_up_test_parameters(config, deadly=True)
    pump_simulation(simulation, duration=pd.Timedelta(days=365))
    pop = simulation.population.population
    dead = pop[pop.alive == 'dead']
    assert len(dead) > 0
    assert np.all(disability.disability_weight(dead.index) == 0)
