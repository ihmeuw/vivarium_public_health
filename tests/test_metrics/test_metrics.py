import numpy as np
import pytest

from datetime import timedelta

from ceam.framework.state_machine import Transition, State

from ceam_tests.util import (setup_simulation, pump_simulation, build_table,
                             generate_test_population)

from ceam_public_health.components.metrics import Metrics
from ceam_public_health.components.disease import (DiseaseState,
                                                   ExcessMortalityState,
                                                   DiseaseModel)

@pytest.fixture
def set_up_test_parameters(flu=False, mumps=False):
    """
    Sets up a simulation with specified disease states

    flu: bool
        If true, include an excess mortality state for flu
        If false, do not include an excess mortality state for flu

    mumps: bool
        If true, include an excess mortality state for mumps
        If false, do not include an excess mortality state for mumps
    """
    healthy_model = DiseaseModel('healthy')
    healthy_state = ExcessMortalityState('healthy', disability_weight=0.0,
                                         excess_mortality_data=build_table(0),
                                         prevalence_data=build_table(1.0, ['age',
                                                                           'year',
                                                                           'sex',
                                                                           'prevalence']),
                                         csmr_data=build_table(0))
    healthy_model.states.extend([healthy_state])

    simulation = setup_simulation([generate_test_population, healthy_model, Metrics()],
                                  population_size=1000)


    if flu:
        flu_model = DiseaseModel('flu')
        flu = ExcessMortalityState('flu', disability_weight=0.2,
                                   excess_mortality_data=build_table(0),
                                   prevalence_data=build_table(1.0,
                                       ['age', 'year', 'sex', 'prevalence']),
                                   csmr_data=build_table(0))

        flu_model.states.extend([flu])
        
        simulation = setup_simulation([generate_test_population, healthy_model, flu_model, 
                                      Metrics()], population_size=1000)

    if mumps:
        mumps_model = DiseaseModel('mumps')
        mumps = ExcessMortalityState('mumps', disability_weight=0.4,
                                     excess_mortality_data=build_table(0),
                                     prevalence_data=build_table(1.0,
                                         ['age', 'year', 'sex', 'prevalence']),
                                     csmr_data=build_table(0))

        mumps_model.states.extend([mumps])

        simulation = setup_simulation([generate_test_population, healthy_model, flu_model,
                                      mumps_model, Metrics()], population_size=1000)


    metrics = simulation.values.get_value('metrics')

    return simulation, metrics


def test_that_ylds_are_0_at_sim_beginning():
    simulation, metrics = set_up_test_parameters()

    assert metrics(simulation.population.population.index)['years_lived_with_disability'] == 0, \
        "at the beginning of the simulation, ylds should = 0"


def test_that_healthy_people_dont_accrue_disability_weights():

    simulation, metrics = set_up_test_parameters()

    pump_simulation(simulation, duration=timedelta(days=365))

    assert np.isclose(metrics(simulation.population.population.index)['years_lived_with_disability'],
                      10000 * 0.0, rtol=0.01), "If everyone is healthy, disability weight should be 0"


def test_single_disability_weight():
    # Flu season
    simulation, metrics = set_up_test_parameters(flu=True)

    pump_simulation(simulation, duration=timedelta(days=365))

    # check that disability weight is correctly calculated
    assert np.isclose(metrics(simulation.population.population.index)['years_lived_with_disability'],
                      1000 * 0.2, rtol=0.01), "YLDs metric should accurately sum up YLDs in the sim"


def test_joint_disability_weight():
    # Flu season in conjunction with a mumps outbreak
    simulation, metrics = set_up_test_parameters(flu=True, mumps=True)

    pump_simulation(simulation, duration=timedelta(days=365))

    # check that JOINT disability weight is correctly calculated
    assert np.isclose(metrics(simulation.population.population.index)['years_lived_with_disability'],
                      1000 * 0.52, rtol=0.01), "YLDs metric should accurately sum up YLDs in the sim"
