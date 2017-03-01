import pytest
import pandas as pd, numpy as np
from ceam import config
from ceam_public_health.components.diarrhea_disease_model import DiarrheaEtiologyState, ApplyDiarrheaExcessMortality, ApplyDiarrheaRemission, diarrhea_factory
from ceam_tests.util import build_table, setup_simulation, generate_test_population, pump_simulation
from ceam_public_health.components.disease import DiseaseModel, RateTransition
from ceam.framework.state_machine import State, Transition
from ceam_inputs import get_etiology_specific_incidence, get_excess_mortality, get_cause_specific_mortality, get_duration_in_days
from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns
from ceam_public_health.components.accrue_susceptible_person_time import AccrueSusceptiblePersonTime
from ceam.framework.event import Event
from datetime import timedelta


list_of_etiologies = ['diarrhea_due_to_shigellosis', 'diarrhea_due_to_cholera', 'diarrhea_due_to_other_salmonella', 'diarrhea_due_to_EPEC', 'diarrhea_due_to_ETEC', 'diarrhea_due_to_campylobacter', 'diarrhea_due_to_amoebiasis', 'diarrhea_due_to_cryptosporidiosis', 'diarrhea_due_to_rotaviral_entiritis', 'diarrhea_due_to_aeromonas', 'diarrhea_due_to_clostridium_difficile', 'diarrhea_due_to_norovirus', 'diarrhea_due_to_adenovirus']


def test_diarrhea_factory():
    factory = diarrhea_factory()

    simulation = setup_simulation([generate_test_population] + factory)

    # make it so that all men will get incidence due to rotaviral entiritis
    inc = build_table(0)

    inc.loc[inc.sex == 'Male', 'rate'] = 10000000

    rota_inc = simulation.values.get_rate('incidence_rate.diarrhea_due_to_rotaviral_entiritis')

    rota_inc.source = simulation.tables.build_table(inc)

    # make it so that all men who get diarrhea over the age of 40 will die, all men under the age of 40 will live
    x_mort = build_table(0)

    x_mort.loc[x_mort.age >= 40, 'rate'] = 1000

    excess_mortality_rate = simulation.values.get_rate('excess_mortality.diarrhea')

    excess_mortality_rate.source = simulation.tables.build_table(x_mort) 

    # pump the simulation forward 1 time period
    pump_simulation(simulation, time_step_days=1, iterations=1)

    only_men = simulation.population.population.query("sex == 'Male'")
    only_women = simulation.population.population.query("sex == 'Female'")

    men_under40 = only_men.query("age <= 39")
    men_over40 = only_men.query("age >= 40")

    # TEST 1 --> test that incidence rate is correctly being applied
    assert simulation.population.population.diarrhea_due_to_rotaviral_entiritis_event_count.sum() == len(only_men), "all men should have diarrhea due to rotavirus after the first timestep in this test"

    # TEST 2 --> test that disability weight is correctly being applied
    dis_weight = simulation.values.get_value('disability_weight')

    # FIXME: This test fails if time step is set to be other than one day in the config file
    assert pd.unique(dis_weight(only_men.index)) == np.array(.2319 * 1/365), "disability weight needs to accurately be set each time step a simulant has diarrhea"
    assert pd.unique(dis_weight(only_women.index)) == np.array(0), "disability should be 0 each time step a simulant does not have diarrhea"

    # TEST 3 --> test excess mortality
    excess_mortality_rate = simulation.values.get_rate('excess_mortality.diarrhea')

    assert pd.unique(excess_mortality_rate(men_over40.index)) == np.array(1000*1/365), "ApplyDiarrheaExcessMortality needs to correctly assign the excess mortality of diarrhea"
    
    # FIXME: Need to figure out why line below is broken test. it should be working, but one simulant appears to have a strange value
    assert pd.unique(excess_mortality_rate(men_under40.index)) == np.array(0), "ApplyDiarrheaExcessMortality needs to correctly assign the excess mortality of diarrhea"


def test_remission():
    factory = diarrhea_factory()

    simulation = setup_simulation([generate_test_population] + factory)
    emitter = simulation.events.get_emitter('time_step')

    # make it so that duration of diarrhea is 1 day among men under 20, but 2 days for men between age 20 and 40
    dur = build_table(1, ['age', 'year', 'sex', 'duration'])

    dur.loc[(dur.age >=20) & (dur.age <40), 'duration'] = 2

    duration = simulation.values.get_value('duration.diarrhea')

    duration.source = simulation.tables.build_table(dur)

    # make it so that all men will get incidence due to rotaviral entiritis
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
    assert pd.unique(diarrhea_first_time_step.diarrhea) == np.array('diarrhea'), "duration should correctly determine the duration of a bout of diarrhea"

    emitter(Event(simulation.population.population.index))

    simulation.current_time += timedelta(days=1)

    # move the simulation forward, make sure that at least some male simulants under 20 remitted into the healthy state (some may not be healthy if they got diarrhea again)
    assert "healthy" in pd.unique(simulation.population.population.loc[diarrhea_first_time_step.index].query("age < 20 and sex == 'Male'").diarrhea), "duration should correctly determine the duration of a bout of diarrhea"

    # make sure that all simulants between the age of 20 and 40 that got diarrhea in the first time step still have diarrhea
    assert "healthy" not in pd.unique(simulation.population.population.loc[diarrhea_first_time_step.index].query("age >= 20 and age < 40 and sex == 'Male'").diarrhea), "duration should correctly determine the duration of a bout of diarrhea"

# End.
