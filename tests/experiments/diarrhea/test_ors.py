import os

import numpy as np

from vivarium import config
from vivarium.test_util import build_table, setup_simulation, generate_test_population, pump_simulation

from ceam_inputs import get_ors_exposures, get_ors_pafs, get_ors_relative_risks, get_severe_diarrhea_excess_mortality

from ceam_public_health.experiments.diarrhea.components.diarrhea import diarrhea_factory
from ceam_public_health.experiments.diarrhea.components.ors import Ors


def setup():
    # Remove user overrides but keep custom cache locations if any
    try:
        config.reset_layer('override', preserve_keys=['input_data.intermediary_data_cache_path',
                                                      'input_data.auxiliary_data_folder'])
    except KeyError:
        pass

    config.simulation_parameters.set_with_metadata('year_start', 2010, layer='override',
                                                   source=os.path.realpath(__file__))
    config.simulation_parameters.set_with_metadata('year_end', 2015, layer='override',
                                                   source=os.path.realpath(__file__))
    config.simulation_parameters.set_with_metadata('time_step', 1, layer='override', source=os.path.realpath(__file__))
    config.simulation_parameters.set_with_metadata('initial_age', 0, layer='override',
                                                   source=os.path.realpath(__file__))
    config.simulation_parameters.set_with_metadata('location_id', 179, layer='override',
                                                   source=os.path.realpath(__file__))
    config.ors.set_with_metadata('ors_exposure_increase_above_baseline', .5, layer='override', source=os.path.realpath(__file__))
    config.ors.set_with_metadata('run_intervention', True, layer='override', source=os.path.realpath(__file__))


def test_determine_who_gets_ors():
    """
    Ensure that the correct number of simulants receive ORS. Take into account\
    the baseline coverage (from GBD) and the effect of any intervention
    """
    factory = diarrhea_factory()

    population_size = 200000

    simulation = setup_simulation([generate_test_population, Ors()] + factory, population_size=population_size)

    # make it so that all men will get diarrhea
    inc = build_table(0)
    inc.loc[inc.sex == 'Male', 'rate'] = 14000
    rota_inc = simulation.values.get_rate('incidence_rate.rotaviral_entiritis')
    rota_inc.source = simulation.tables.build_table(inc)

    pump_simulation(simulation, iterations=1)

    pop = simulation.population.population
    males_with_diarrhea = pop.query("sex == 'Male' and diarrhea != 'healthy'")
    ors_proportion = len(males_with_diarrhea.query('ors_working == 1'))/len(males_with_diarrhea)
    no_ors_proportion = len(males_with_diarrhea.query('ors_working == 0')) / len(males_with_diarrhea)

    ors_exposure = get_ors_exposures()

    GBD_proportion_exposed = ors_exposure.query(
        "sex == 'Male' and age <.01").set_index(['year']).get_value(2010, 'cat2')
    GBD_proportion_not_exposed = ors_exposure.query(
        "sex == 'Male' and age <.01").set_index(['year']).get_value(2010, 'cat1')

    assert np.allclose(ors_proportion, (GBD_proportion_exposed +
                       config.ors.ors_exposure_increase_above_baseline),
                       rtol=.05), \
        "proportion of people on ors should accurately reflect the exposure from GBD"

    assert np.allclose(no_ors_proportion, (GBD_proportion_not_exposed -
                       config.ors.ors_exposure_increase_above_baseline),
                       rtol=.05), \
        "proportion of people on ors should accurately reflect the exposure from GBD"


def test_ors_working_column():
    """
    Test that the ors working column is only set for people that should receive
    ORS and only in the correct time window
    """
    factory = diarrhea_factory()
    simulation = setup_simulation([generate_test_population, Ors()] + factory)
    pump_simulation(simulation, iterations=1)
    pop = simulation.population.population
    ors_pop = pop.query("ors_working == 1")

    assert (ors_pop.diarrhea != 'healthy').all(), \
            "assert that all people who receive ors also have diarrhea"

def test_mortality_rates():
    """
    Test that people who are unexposed to ORS and have severe diarrhea have the
    severe diarrhea mortality rate * (1 - PAF) and people that have severe
    diarrhea and get ORS have the severe diarrhea mortality rate * (1 - PAF) * rr
    """
    factory = diarrhea_factory()
    simulation = setup_simulation([generate_test_population, Ors()] + factory)
    # make it so that all men will get diarrhea
    inc = build_table(0)
    inc.loc[inc.sex == 'Male', 'rate'] = 140000
    rota_inc = simulation.values.get_rate('incidence_rate.rotaviral_entiritis')
    rota_inc.source = simulation.tables.build_table(inc)

    pump_simulation(simulation, iterations=1)

    excess_mortality_rate = simulation.values.get_rate('excess_mortality.diarrhea')
    pop = simulation.population.population.query("sex == 'Male'")
    ors_pop = pop.query("ors_working == 1")
    diarrhea_but_no_ors_pop = pop.query("ors_working == 0 and diarrhea != 'healthy'")

    pafs = get_ors_pafs()
    GBD_paf = pafs.query("sex == 'Male' and age <.01").set_index(['year']).get_value(2010, 'paf')

    GBD_rr = get_ors_relative_risks()

    mr = get_severe_diarrhea_excess_mortality()
    GBD_mr = mr.query("sex == 'Male' and age <.01").set_index(['year']).get_value(2010, 'rate')

    no_ors_diarrhea_mortality_rate = GBD_mr * (1 - GBD_paf) * GBD_rr * 1/365
    ors_diarrhea_mortality_rate = GBD_mr * (1 - GBD_paf) * 1/365

    simulation_no_ors_rate = excess_mortality_rate(diarrhea_but_no_ors_pop.index).unique()
    simulation_ors_rate = excess_mortality_rate(ors_pop.index).unique()

    assert np.allclose(no_ors_diarrhea_mortality_rate, simulation_no_ors_rate, rtol=.05)

    assert np.allclose(ors_diarrhea_mortality_rate, simulation_ors_rate, rtol=.05)
