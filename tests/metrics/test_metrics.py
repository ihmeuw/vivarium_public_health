import os
from datetime import timedelta

import numpy as np

from ceam import config
from ceam_tests.util import setup_simulation, pump_simulation, build_table, generate_test_population

from ceam_public_health.disease import ExcessMortalityState, DiseaseModel
from ceam_public_health.metrics import Metrics


def setup():
    try:
        config.reset_layer('override', preserve_keys=['input_data.intermediary_data_cache_path',
                                                      'input_data.auxiliary_data_folder'])
    except KeyError:
        pass
    config.simulation_parameters.set_with_metadata('year_start', 1990, layer='override',
                                                   source=os.path.realpath(__file__))
    config.simulation_parameters.set_with_metadata('year_end', 2000, layer='override',
                                                   source=os.path.realpath(__file__))
    config.simulation_parameters.set_with_metadata('time_step', 30.5, layer='override',
                                                   source=os.path.realpath(__file__))


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
    # first start with an asymptomatic disease state. asymptomatic
    #     diseases have 0 disability weight in GBD.
    asymptomatic_disease_model = DiseaseModel('asymptomatic')
    asymptomatic_disease_state = ExcessMortalityState('asymptomatic',
                                                      disability_weight=0.0,
                                                      excess_mortality_data=build_table(0),
                                                      prevalence_data=build_table(1.0,
                                                        ['age', 'year', 'sex',
                                                         'prevalence']),
                                                      csmr_data=build_table(0))

    asymptomatic_disease_model.states.extend([asymptomatic_disease_state])

    if not mumps and not flu:
        simulation = setup_simulation([generate_test_population,
                                       asymptomatic_disease_model, Metrics()],
                                      population_size=1000)

    # Now let's set up a disease model for a disease that does have
    #     a disability weight
    if flu:
        flu_model = DiseaseModel('flu')
        flu = ExcessMortalityState('flu', disability_weight=0.2,
                                   excess_mortality_data=build_table(0),
                                   prevalence_data=build_table(1.0,
                                       ['age', 'year', 'sex', 'prevalence']),
                                   csmr_data=build_table(0))

        flu_model.states.extend([flu])

        if not mumps:
            simulation = setup_simulation([generate_test_population,
                                           asymptomatic_disease_model, flu_model,
                                          Metrics()], population_size=1000)

    # Now let's set up another disease model so we can test that
    #     CEAM is calculating joint disability weights correctly
    if mumps:
        mumps_model = DiseaseModel('mumps')
        mumps = ExcessMortalityState('mumps', disability_weight=0.4,
                                     excess_mortality_data=build_table(0),
                                     prevalence_data=build_table(1.0,
                                         ['age', 'year', 'sex', 'prevalence']),
                                     csmr_data=build_table(0))

        mumps_model.states.extend([mumps])

        simulation = setup_simulation([generate_test_population,
                                       asymptomatic_disease_model, flu_model,
                                      mumps_model, Metrics()],
                                      population_size=1000)

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
                      10000 * 0.0, rtol=0.01), "If no one has a disabling" + \
                                               " disease, YLDs should be 0"


def test_single_disability_weight():
    # Flu season
    simulation, metrics = set_up_test_parameters(flu=True)

    pump_simulation(simulation, duration=timedelta(days=365))

    # check that disability weight is correctly calculated
    assert np.isclose(metrics(simulation.population.population.index)['years_lived_with_disability'],
                      1000 * 0.2, rtol=0.01), "YLDs metric should accurately" + \
                                              " sum up YLDs in the sim." + \
                                              " In this case, all simulants" + \
                                              " should accrue a disability" + \
                                              " weight of .2 since all simulants" + \
                                              " have the flu for the" + \
                                              " entire year and the disability weight" + \
                                              " of the flu is .2"


def test_joint_disability_weight():
    # Flu season in conjunction with a mumps outbreak
    simulation, metrics = set_up_test_parameters(flu=True, mumps=True)

    pump_simulation(simulation, duration=timedelta(days=365))

    # check that JOINT disability weight is correctly calculated
    assert np.isclose(metrics(simulation.population.population.index)['years_lived_with_disability'],
                      1000 * (1-(1-.2)*(1-.4)), rtol=0.01), "YLDs metric should accurately" + \
                                               " sum up YLDs in the sim." + \
                                               " In this case, all simulants" + \
                                               " should accrue a disability" + \
                                               " weight of .52 since all" + \
                                               " simulants have the flu and mumps" + \
                                               " for the entire year." + \
                                               " The disability weight of the flu is .2" + \
                                               " and the disability weight" +\
                                               " of mumps is .4. Thus, each" + \
                                               " simulant should have a" + \
                                               " disability weight of .52 for" + \
                                               " this year given that joint disability equals" + \
                                               " 1 - (1 - dis wt 1) * (1 - dis wt 2)... * (1 - dis wt i)"

