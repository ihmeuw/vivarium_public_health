import pytest
import pandas as pd, numpy as np
from ceam import config
from ceam_public_health.components.diarrhea_disease_model import DiarrheaEtiologyState, DiarrheaBurden, diarrhea_factory
from ceam_tests.util import build_table, setup_simulation, generate_test_population, pump_simulation
from ceam_public_health.components.disease import DiseaseModel, RateTransition
from ceam.framework.state_machine import State, Transition
from ceam_inputs import get_etiology_specific_incidence, get_excess_mortality, get_cause_specific_mortality, get_duration_in_days, get_cause_deleted_mortality_rate, get_cause_specific_mortality
from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns
from ceam_public_health.components.accrue_susceptible_person_time import AccrueSusceptiblePersonTime
from ceam.framework.event import Event
from datetime import timedelta, datetime
from ceam_public_health.components.base_population import Mortality
from ceam_inputs.gbd_ms_functions import get_disability_weight
from ceam_inputs import get_severity_splits

def make_simulation_object():
    factory = diarrhea_factory()

    simulation = setup_simulation([generate_test_population] + factory, start=datetime(2005, 1, 1))

    # make it so that all men will get incidence due to rotaviral entiritis
    inc = build_table(0)

    inc.loc[inc.sex == 'Male', 'rate'] = 14000

    rota_inc = simulation.values.get_rate('incidence_rate.diarrhea_due_to_rotaviral_entiritis')

    rota_inc.source = simulation.tables.build_table(inc)

    # make it so that all men who get diarrhea over the age of 40 have a high excess mortality, all men under the age of 40 do not
    x_mort = build_table(0)

    # TODO: Figure out the exact rate needed to get a probability of 1 when timestep is 1
    x_mort.loc[x_mort.age >= 40, 'rate'] = 14000

    excess_mortality_rate = simulation.values.get_rate('excess_mortality.diarrhea')

    excess_mortality_rate.source = simulation.tables.build_table(x_mort) 

    return simulation

# TEST 1 --> test that incidence rate is correctly being applied
def test_incidence_rates():

    simulation = make_simulation_object()

    # pump the simulation forward 1 time period
    pump_simulation(simulation, time_step_days=1, iterations=1, year_start=2005)

    only_men = simulation.population.population.query("sex == 'Male'")

    assert simulation.population.population.diarrhea_due_to_rotaviral_entiritis_event_count.sum() == len(only_men), "all men should have diarrhea due to rotavirus after the first timestep in this test"

# TEST 2 --> test that disability weight is correctly being applied
def test_disability_weights():
    simulation = make_simulation_object()

    dis_weight = simulation.values.get_value('disability_weight')

    # pump the simulation forward 1 time period
    pump_simulation(simulation, time_step_days=1, iterations=1, year_start=2005)

    ts = config.getint('simulation_parameters', 'time_step')
    mild_disability_weight = get_disability_weight(healthstate_id=355)*ts/365
    moderate_disability_weight = get_disability_weight(healthstate_id=356)*ts/365
    severe_disability_weight = get_disability_weight(healthstate_id=357)*ts/365

    # TEST 2A --> Check that there are no unexpected disability weights
    only_men = simulation.population.population.query("sex == 'Male'")
    assert np.allclose(np.sort(pd.unique(dis_weight(only_men.index))), np.sort([mild_disability_weight, moderate_disability_weight, severe_disability_weight])), "assert that disability weights values are what they are expected to be"

    # TEST 2B --> Check that the disability weights are mapped correctly
    mild_diarrhea_index = simulation.population.population.query("diarrhea == 'mild_diarrhea'").index
    moderate_diarrhea_index = simulation.population.population.query("diarrhea == 'moderate_diarrhea'").index
    severe_diarrhea_index = simulation.population.population.query("diarrhea == 'severe_diarrhea'").index
    assert np.allclose(pd.unique(dis_weight(mild_diarrhea_index)), np.array([mild_disability_weight])), "diarrhea severity state should be correctly mapped to its specific disability weight"
    assert np.allclose(pd.unique(dis_weight(moderate_diarrhea_index)), np.array([moderate_disability_weight])), "diarrhea severity state should be correctly mapped to its specific disability weight"
    assert np.allclose(pd.unique(dis_weight(severe_diarrhea_index)), np.array([severe_disability_weight])), "diarrhea severity state should be correctly mapped to its specific disability weight"

# TEST 3 --> test excess mortality
def test_excess_mortality():
    simulation = make_simulation_object()

    # pump the simulation forward 1 time period
    pump_simulation(simulation, time_step_days=1, iterations=1, year_start=2005)

    excess_mortality_rate = simulation.values.get_rate('excess_mortality.diarrhea')

    ts = config.getint('simulation_parameters', 'time_step')
    only_men = simulation.population.population.query("sex == 'Male'")
    men_under40 = only_men.query("age <= 39")
    men_over40 = only_men.query("age >= 40")

    assert pd.unique(excess_mortality_rate(men_over40.index)) == np.array(14000*ts/365), "DiarrheaBurden needs to correctly assign the excess mortality of diarrhea"
    
    assert pd.unique(excess_mortality_rate(men_under40.index)) == np.array(0), "DiarrheaBurden needs to correctly assign the excess mortality of diarrhea"

# TEST 4 --> test remission
def test_remission():
    factory = diarrhea_factory()

    simulation = setup_simulation([generate_test_population] + factory, start=datetime(2005, 1, 1))
    emitter = simulation.events.get_emitter('time_step')

    # make it so that duration of diarrhea is 1 day among all men except for men between age 20 and 40, for whom duration will be 2 days
    dur = build_table(1, ['age', 'year', 'sex', 'duration'])

    dur.loc[(dur.age >=20) & (dur.age <40), 'duration'] = 2

    duration = simulation.values.get_value('duration.diarrhea')

    duration.source = simulation.tables.build_table(dur)

    # make it so that some men will get diarrhea due to rota
    inc = build_table(0)

    inc.loc[inc.sex == 'Male', 'rate'] = 200

    rota_inc = simulation.values.get_rate('incidence_rate.diarrhea_due_to_rotaviral_entiritis')

    rota_inc.source = simulation.tables.build_table(inc)

    # pump_simulation(simulation, time_step_days=2, iterations=1)    
    # Move everyone into the event state
    emitter(Event(simulation.population.population.index))

    simulation.current_time += timedelta(days=1)

    pop = simulation.population.population

    diarrhea_first_time_step = pop[pop.diarrhea_event_time.notnull()]

    # check that everyone has been moved into the diarrhea state
    assert set(pd.unique(diarrhea_first_time_step.diarrhea)) == set(['mild_diarrhea', 'moderate_diarrhea', 'severe_diarrhea']), "duration should correctly determine the duration of a bout of diarrhea"

    emitter(Event(simulation.population.population.index))

    simulation.current_time += timedelta(days=1)

    # move the simulation forward, make sure that at least some male simulants under 20 remitted into the healthy state (some may not be healthy if they got diarrhea again)
    assert "healthy" in pd.unique(simulation.population.population.loc[diarrhea_first_time_step.index].query("age < 20 and sex == 'Male'").diarrhea), "duration should correctly determine the duration of a bout of diarrhea"

    # make sure that all simulants between the age of 20 and 40 that got diarrhea in the first time step still have diarrhea
    assert "healthy" not in pd.unique(simulation.population.population.loc[diarrhea_first_time_step.index].query("age >= 20 and age < 40 and sex == 'Male'").diarrhea), "duration should correctly determine the duration of a bout of diarrhea"

# TEST 5 --> test that severe_diarrhea is the only severity level of diarrhea associated with an excess mortality
def test_diarrhea_elevated_mortality():
    factory = diarrhea_factory()

    simulation = setup_simulation([generate_test_population] + factory, start=datetime(2005, 1, 1))

    # make it so that all men will get incidence due to rotaviral entiritis
    inc = build_table(0)

    inc.loc[inc.sex == 'Male', 'rate'] = 14000

    rota_inc = simulation.values.get_rate('incidence_rate.diarrhea_due_to_rotaviral_entiritis')

    rota_inc.source = simulation.tables.build_table(inc)

    # make the base mortality_rate 0
    mortality_rate = simulation.values.get_rate('mortality_rate')

    mortality_rate.source = simulation.tables.build_table(build_table(0))
 
    pump_simulation(simulation, time_step_days=1, iterations=1, year_start=2005)

    pop = simulation.population.population

    severe_diarrhea_index = pop.query("diarrhea == 'severe_diarrhea'").index

    excess_mortality_rate = simulation.values.get_rate('mortality_rate')

    # FIXME: @Alecwd -- is this the correct use of .all()? Are there better ways to check that all values are greater than 0? Similar question applies to two tests below.
    assert excess_mortality_rate(severe_diarrhea_index)['death_due_to_severe_diarrhea'].all() > 0, "people with diarrhea should have an elevated mortality rate"

    mild_diarrhea_index = pop.query("diarrhea == 'mild_diarrhea'").index
    moderate_diarrhea_index = pop.query("diarrhea == 'moderate_diarrhea'").index

    assert excess_mortality_rate(mild_diarrhea_index).all() == 0, "people with mild/moderate diarrhea should have no elevated mortality due to diarrhea (or due to anything else in this test)"
    assert excess_mortality_rate(moderate_diarrhea_index).all() == 0, "people with mild/moderate diarrhea should have no elevated mortality due to diarrhea (or due to anything else in this test)"

# TEST 6 --> test that severity proportions are correctly being applied
def test_severity_proportions():
    factory = diarrhea_factory()

    simulation = setup_simulation([generate_test_population] + factory, start=datetime(2005, 1, 1), population_size=1000)

    # give everyone diarrhea
    inc = build_table(14000)

    rota_inc = simulation.values.get_rate('incidence_rate.diarrhea_due_to_rotaviral_entiritis')

    rota_inc.source = simulation.tables.build_table(inc)

    # pump the simulation forward 1 time period
    pump_simulation(simulation, time_step_days=1, iterations=1, year_start=2005)

    pop = simulation.population.population

    mild_prop_in_sim = len(pop.query("diarrhea == 'mild_diarrhea'"))/1000

    moderate_prop_in_sim = len(pop.query("diarrhea == 'moderate_diarrhea'"))/1000

    severe_prop_in_sim = len(pop.query("diarrhea == 'severe_diarrhea'"))/1000

    severe_prop_in_GBD = get_severity_splits(1181, 2610)

    moderate_prop_in_GBD = get_severity_splits(1181, 2609)

    mild_prop_in_GBD = get_severity_splits(1181, 2608)

    assert np.allclose(mild_prop_in_sim, mild_prop_in_GBD, atol=.01)

    assert np.allclose(moderate_prop_in_sim, moderate_prop_in_GBD, atol=.01)

    assert np.allclose(severe_prop_in_sim, severe_prop_in_GBD, atol=.01)

# End.
