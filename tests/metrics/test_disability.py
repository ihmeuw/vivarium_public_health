import os

import numpy as np
import pandas as pd

from vivarium import config
from vivarium.test_util import setup_simulation, pump_simulation, build_table, generate_test_population

from ceam_public_health.population import Mortality
from ceam_public_health.disease import ExcessMortalityState, DiseaseModel
from ceam_public_health.metrics import Metrics, Disability


def setup():
    try:
        config.reset_layer('override', preserve_keys=['input_data.intermediary_data_cache_path',
                                                      'input_data.auxiliary_data_folder'])
    except KeyError:
        pass
    config.simulation_parameters.set_with_metadata('year_start', 1995, layer='override',
                                                   source=os.path.realpath(__file__))
    config.simulation_parameters.set_with_metadata('year_end', 2000, layer='override',
                                                   source=os.path.realpath(__file__))
    config.simulation_parameters.set_with_metadata('time_step', 30.5, layer='override',
                                                   source=os.path.realpath(__file__))


def set_up_test_parameters(flu=False, mumps=False, deadly=False):
    """
    Sets up a simulation with specified disease states

    flu: bool
        If true, include an excess mortality state for flu
        If false, do not include an excess mortality state for flu

    mumps: bool
        If true, include an excess mortality state for mumps
        If false, do not include an excess mortality state for mumps
    """
    n_simulants = 1000
    asymptomatic_disease_state = ExcessMortalityState('asymptomatic',
                                                      disability_weight=0.0,
                                                      excess_mortality_data=build_table(0),
                                                      prevalence_data=build_table(1.0,
                                                                                  ['age', 'year', 'sex', 'prevalence']))
    asymptomatic_disease_model = DiseaseModel('asymptomatic',
                                              states=[asymptomatic_disease_state],
                                              csmr_data=build_table(0))
    metrics = Metrics()
    disability = Disability()
    components = [generate_test_population, asymptomatic_disease_model, metrics, disability]

    if flu:
        flu = ExcessMortalityState('flu', disability_weight=0.2,
                                   excess_mortality_data=build_table(0),
                                   prevalence_data=build_table(1.0,
                                                               ['age', 'year', 'sex', 'prevalence']))
        flu_model = DiseaseModel('flu', states=[flu], csmr_data=build_table(0))
        components.append(flu_model)

    if mumps:
        mumps = ExcessMortalityState('mumps', disability_weight=0.4,
                                     excess_mortality_data=build_table(0),
                                     prevalence_data=build_table(1.0,
                                                                 ['age', 'year', 'sex', 'prevalence']))
        mumps_model = DiseaseModel('mumps', states=[mumps], csmr_data=build_table(0))
        components.append(mumps_model)

    if deadly:
        deadly = ExcessMortalityState('deadly', disability_weight=0.4,
                                     excess_mortality_data=build_table(1),
                                     prevalence_data=build_table(1.0,
                                                                 ['age', 'year', 'sex', 'prevalence']))
        deadly_model = DiseaseModel('deadly', states=[deadly], csmr_data=build_table(0.05))
        components.append(deadly_model)
        components.append(Mortality())

    simulation = setup_simulation(components=components, population_size=n_simulants)

    return simulation, metrics, disability


def test_that_ylds_are_0_at_sim_beginning():
    simulation, metrics, disability = set_up_test_parameters()
    ylds = metrics.metrics(simulation.population.population.index)['years_lived_with_disability']
    assert ylds == 0


def test_that_healthy_people_dont_accrue_disability_weights():
    simulation, metrics, disability = set_up_test_parameters()
    pump_simulation(simulation, duration=pd.Timedelta(days=365))
    pop_size = len(simulation.population.population)
    ylds = metrics.metrics(simulation.population.population.index)['years_lived_with_disability']
    assert np.isclose(ylds, pop_size * 0.0, rtol=0.01)


def test_single_disability_weight():
    simulation, metrics, disability = set_up_test_parameters(flu=True)
    flu_dw = 0.2
    pump_simulation(simulation, duration=pd.Timedelta(days=365))
    pop_size = len(simulation.population.population)
    ylds = metrics.metrics(simulation.population.population.index)['years_lived_with_disability']
    assert np.isclose(ylds, pop_size * flu_dw, rtol=0.01)


def test_joint_disability_weight():
    simulation, metrics, disability = set_up_test_parameters(flu=True, mumps=True)
    flu_dw = 0.2
    mumps_dw = 0.4
    pump_simulation(simulation, duration=pd.Timedelta(days=365))
    pop_size = len(simulation.population.population)
    ylds = metrics.metrics(simulation.population.population.index)['years_lived_with_disability']
    # check that JOINT disability weight is correctly calculated
    assert np.isclose(ylds, pop_size * (1-(1-flu_dw)*(1-mumps_dw)), rtol=0.01)


def test_dead_people_dont_accrue_disability():
    simulation, metrics, disability = set_up_test_parameters(deadly=True)
    pump_simulation(simulation, duration=pd.Timedelta(days=365))
    pop = simulation.population.population
    dead = pop[pop.alive == 'dead']
    assert len(dead) > 0
    assert np.all(disability.disability_weight(dead.index) == 0)

