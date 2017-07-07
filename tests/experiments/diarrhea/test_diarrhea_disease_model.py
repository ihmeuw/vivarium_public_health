import os
from datetime import timedelta

import numpy as np
import pandas as pd

from ceam import config
from ceam.framework.event import Event
from ceam.test_util import (build_table, setup_simulation,
                             generate_test_population, pump_simulation)

from ceam_inputs import (get_severity_splits, get_cause_specific_mortality,
                         get_cause_deleted_mortality_rate, get_disability_weight, causes)

from ceam_public_health.metrics import Disability
from ceam_public_health.population import Mortality

from ceam_public_health.experiments.diarrhea.components.diarrhea import diarrhea_factory



def setup():
    try:
        config.reset_layer('override', preserve_keys=['input_data.intermediary_data_cache_path',
                                                      'input_data.auxiliary_data_folder'])
    except KeyError:
        pass
    config.simulation_parameters.set_with_metadata('year_start', 2005, layer='override',
                                                   source=os.path.realpath(__file__))
    config.simulation_parameters.set_with_metadata('year_end', 2010, layer='override',
                                                   source=os.path.realpath(__file__))
    config.simulation_parameters.set_with_metadata('time_step', 1, layer='override',
                                                   source=os.path.realpath(__file__))
    config.simulation_parameters.set_with_metadata('initial_age', None, layer='override',
                                                   source=os.path.realpath(__file__))
    config.simulation_parameters.set_with_metadata('num_simulants', 1000, layer='override',
                                                   source=os.path.realpath(__file__))

def make_simulation_object():
    factory = diarrhea_factory()

    simulation = setup_simulation([generate_test_population, Disability()] + factory)

    # make it so that all men will get incidence due to rotaviral entiritis
    inc = build_table(0)

    inc.loc[inc.sex == 'Male', 'rate'] = 14000

    rota_inc = simulation.values.get_rate('incidence_rate.rotaviral_entiritis')

    rota_inc.source = simulation.tables.build_table(inc)

    # make it so that all men who get diarrhea over the age of 40 have a high
    #     excess mortality, all men under the age of 40 do not
    x_mort = build_table(0)

    # TODO: Figure out the exact rate needed to get a probability of 1 when
    #     timestep is 1
    x_mort.loc[x_mort.age >= 40, 'rate'] = 14000

    excess_mortality_rate = simulation.values.get_rate('excess_mortality.diarrhea')

    excess_mortality_rate.source = simulation.tables.build_table(x_mort)

    return simulation


# TEST 1 --> test that incidence rate is correctly being applied
def test_incidence_rates():

    simulation = make_simulation_object()

    # pump the simulation forward 1 time period
    pump_simulation(simulation, iterations=1)

    only_men = simulation.population.population.query("sex == 'Male'")

    assert simulation.population.population.rotaviral_entiritis_event_count.sum() == len(only_men), "all men should have diarrhea due to rotavirus after the first timestep in this test"


# TEST 2 --> test that disability weight is correctly being applied
def test_disability_weights():
    simulation = make_simulation_object()
    dis_weight = simulation.values.get_value('disability_weight')

    # pump the simulation forward 1 time period
    pump_simulation(simulation, iterations=1)

    ts = config.simulation_parameters.time_step
    mild_disability_weight = get_disability_weight(healthstate_id=355)*ts/365
    moderate_disability_weight = get_disability_weight(healthstate_id=356)*ts/365
    severe_disability_weight = get_disability_weight(healthstate_id=357)*ts/365

    # TEST 2A --> Check that there are no unexpected disability weights
    only_men = simulation.population.population.query("sex == 'Male'")
    simulation_weights = np.sort(pd.unique(dis_weight(only_men.index)))
    GBD_weights = np.sort([mild_disability_weight, moderate_disability_weight,
                           severe_disability_weight])

    assert np.allclose(simulation_weights, GBD_weights), \
        "assert that disability weights values are what they are expected to be"

    # TEST 2B --> Check that the disability weights are mapped correctly
    mild_diarrhea_index = simulation.population.population.query("diarrhea == 'mild_diarrhea'").index
    moderate_diarrhea_index = simulation.population.population.query("diarrhea == 'moderate_diarrhea'").index
    severe_diarrhea_index = simulation.population.population.query("diarrhea == 'severe_diarrhea'").index

    assert np.allclose(pd.unique(dis_weight(mild_diarrhea_index)), mild_disability_weight), \
        "diarrhea severity state should be correctly mapped to its specific disability weight"
    assert np.allclose(pd.unique(dis_weight(moderate_diarrhea_index)), moderate_disability_weight), \
        "diarrhea severity state should be correctly mapped to its specific disability weight"
    assert np.allclose(pd.unique(dis_weight(severe_diarrhea_index)), severe_disability_weight), \
        "diarrhea severity state should be correctly mapped to its specific disability weight"


# TEST 3 --> test remission
def test_remission():
    factory = diarrhea_factory()

    simulation = setup_simulation([generate_test_population] + factory, population_size=1000)
    emitter = simulation.events.get_emitter('time_step')

    # make it so that duration of diarrhea is 1 day among all men except for
    #     men over age 20, for whom duration will be 2 days
    dur = build_table(1, ['age', 'year', 'sex', 'duration'])

    dur.loc[dur.age >= 20, 'duration'] = 2

    duration = simulation.values.get_value('duration.diarrhea')

    duration.source = simulation.tables.build_table(dur)

    # make it so that some men will get diarrhea due to rota
    inc = build_table(0)

    inc.loc[inc.sex == 'Male', 'rate'] = 200

    rota_inc = simulation.values.get_rate('incidence_rate.rotaviral_entiritis')

    rota_inc.source = simulation.tables.build_table(inc)

    # we need two separate time steps, so we use the event emitter instead of
    #     pump simulation because we need two separate time steps
    emitter(Event(simulation.population.population.index))

    simulation.current_time += timedelta(days=1)

    pop = simulation.population.population

    diarrhea_first_time_step = pop[pop.diarrhea_event_time.notnull()]

    # check that everyone has been moved into the diarrhea state
    assert set(pd.unique(diarrhea_first_time_step.diarrhea)) == \
        set(['mild_diarrhea', 'moderate_diarrhea', 'severe_diarrhea']), \
        "duration should correctly determine the duration of a bout of diarrhea"

    emitter(Event(simulation.population.population.index))

    simulation.current_time += timedelta(days=1)

    # Make sure that at least some male simulants under 20 remitted into the
    #     healthy state (some may not be healthy if they got diarrhea again)
    assert "healthy" in pd.unique(simulation.population.population.loc[diarrhea_first_time_step.index].query("age < 20 and sex == 'Male'").diarrhea), \
        "DiarrheaBurden class should correctly determine the duration of a bout of diarrhea"

    # make sure that all simulants between the over age 20 that got diarrhea in
    #     the first time step still have diarrhea
    assert "healthy" not in pd.unique(simulation.population.population.loc[diarrhea_first_time_step.index].query("age >= 20 and sex == 'Male'").diarrhea), \
        "DiarrheaBurden class should correctly determine the duration of a bout of diarrhea"


# TEST 4 --> test that severe_diarrhea is the only severity level of diarrhea
#     associated with an excess mortality
def test_diarrhea_elevated_mortality():
    factory = diarrhea_factory()

    simulation = setup_simulation([generate_test_population] + factory)

    # make it so that all men will get incidence due to rotaviral entiritis
    inc = build_table(0)

    inc.loc[inc.sex == 'Male', 'rate'] = 14000

    rota_inc = simulation.values.get_rate('incidence_rate.rotaviral_entiritis')

    rota_inc.source = simulation.tables.build_table(inc)

    # make the base mortality_rate 0
    mortality_rate = simulation.values.get_rate('mortality_rate')

    mortality_rate.source = simulation.tables.build_table(build_table(0))

    pump_simulation(simulation, iterations=1)

    pop = simulation.population.population

    severe_diarrhea_index = pop.query("diarrhea == 'severe_diarrhea'").index

    # FIXME: @Alecwd -- is this the correct use of .all()? Are there better
    #     ways to check that all values are greater than 0? Similar question
    #     applies to two tests below.
    assert mortality_rate(severe_diarrhea_index)['death_due_to_severe_diarrhea'].all() > 0, \
        "people with diarrhea should have an elevated mortality rate"

    mild_diarrhea_index = pop.query("diarrhea == 'mild_diarrhea'").index
    moderate_diarrhea_index = pop.query("diarrhea == 'moderate_diarrhea'").index

    assert mortality_rate(mild_diarrhea_index).all() == 0, \
        "people with mild/moderate diarrhea should have no elevated" \
        " mortality due to diarrhea (or due to anything else in this test)"
    assert mortality_rate(moderate_diarrhea_index).all() == 0, \
        "people with mild/moderate diarrhea should have no elevated" \
        " mortality due to diarrhea (or due to anything else in this test)"


# TEST 5 --> test that severity proportions are correctly being applied
def test_severity_proportions():
    factory = diarrhea_factory()

    simulation = setup_simulation([generate_test_population] + factory,
                                  population_size=10000)

    # give everyone diarrhea
    inc = build_table(14000)

    rota_inc = simulation.values.get_rate('incidence_rate.rotaviral_entiritis')

    rota_inc.source = simulation.tables.build_table(inc)

    # pump the simulation forward 1 time period
    pump_simulation(simulation, iterations=1)

    pop = simulation.population.population

    mild_prop_in_sim = len(pop.query("diarrhea == 'mild_diarrhea'"))/10000

    moderate_prop_in_sim = len(pop.query("diarrhea == 'moderate_diarrhea'"))/10000

    severe_prop_in_sim = len(pop.query("diarrhea == 'severe_diarrhea'"))/10000

    severe_prop_in_GBD = get_severity_splits(1181, 2610)

    moderate_prop_in_GBD = get_severity_splits(1181, 2609)

    mild_prop_in_GBD = get_severity_splits(1181, 2608)

    assert np.allclose(mild_prop_in_sim, mild_prop_in_GBD, atol=.01)

    assert np.allclose(moderate_prop_in_sim, moderate_prop_in_GBD, atol=.01)

    assert np.allclose(severe_prop_in_sim, severe_prop_in_GBD, atol=.01)


# TEST 6 -- test that diarrhea csmr is deleted from the background mortality
#     rate
def test_cause_deletion():
    config.simulation_parameters.set_with_metadata('initial_age', 0, layer='override',
                                                   source=os.path.realpath(__file__))

    factory = diarrhea_factory()
    simulation = setup_simulation([generate_test_population, Mortality()] + \
                                  factory)

    # determine what the cause-deleted mortality rate should be
    cause_deleted_mr = get_cause_deleted_mortality_rate([get_cause_specific_mortality(causes.diarrhea.gbd_cause)])

    # get the mortality rate from the simulation
    simulation_mortality_rate = simulation.values.get_rate('mortality_rate')

    # compare for the earliest age group (this test requires that
    #     generate_test_population is set to create a cohort of newborns)
    cause_deleted_mr_values = cause_deleted_mr.query("year==2005 and age<.01").cause_deleted_mortality_rate.values
    simulation_values = simulation_mortality_rate(simulation.population.population.index).death_due_to_other_causes.unique()

    ts = config.simulation_parameters.time_step

    # check that the value in the simulation is what it should be
    # @Alecwd: I don't like how I have to specify an absolute tolerance when I
    #     use np.allclose here. Even though the numbers that I want to compare
    #     are really close, I'm concerned because the assertion that the two
    #     parameters are equal fails without the atol parameter being specified
    #     as below. Is there a better way to confirm that the numbers are only
    #     different because of floating point error and not something that we
    #     need to be concerned about?
    assert np.allclose(np.sort(simulation_values), np.sort(cause_deleted_mr_values * ts/365), atol=.000001), \
        "make sure diarrhea has been deleted from the background mortality rate"

