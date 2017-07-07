import os

import numpy as np
import pandas as pd

from ceam import config
from ceam.test_util import build_table, setup_simulation, generate_test_population, pump_simulation

from ceam_inputs import (get_severity_splits, get_cause_specific_mortality,
                         get_cause_deleted_mortality_rate, get_disability_weight, causes)

from ceam_public_health.metrics import Disability
from ceam_public_health.population import Mortality

from ceam_public_health.experiments.diarrhea.components.diarrhea2 import build_diarrhea_model



def setup():
    # Remove user overrides but keep custom cache locations if any
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
    config.simulation_parameters.set_with_metadata('num_simulants', 1000, layer='override',
                                                   source=os.path.realpath(__file__))
    return config

def make_simulation_object():
    diarrhea_and_etiology_models = build_diarrhea_model()
    simulation = setup_simulation([generate_test_population, Disability()] + diarrhea_and_etiology_models)

    # make it so that all men will get incidence due to rotaviral entiritis
    incidence = build_table(0)
    incidence.loc[incidence.sex == 'Male', 'rate'] = 14000
    rota_incidence = simulation.values.get_rate('incidence_rate.rotaviral_entiritis')
    rota_incidence.source = simulation.tables.build_table(incidence)

    # make it so that all men who get diarrhea over the age of 40 have a high
    # excess mortality, all men under the age of 40 do notquit
    excess_mortality = build_table(0)
    excess_mortality.loc[excess_mortality.age >= 40, 'rate'] = 14000
    excess_mortality_rate = simulation.values.get_rate('excess_mortality.diarrhea')
    excess_mortality_rate.source = simulation.tables.build_table(excess_mortality)

    return simulation


def test_incidence_rates():
    simulation = make_simulation_object()
    pump_simulation(simulation, iterations=1)
    only_men = simulation.population.population.query("sex == 'Male'")
    err_msg = "All men should have diarrhea due to rotavirus after the first timestep in this test."
    assert simulation.population.population.rotaviral_entiritis_event_count.sum() == len(only_men), err_msg
    assert sum(simulation.population.population.rotaviral_entiritis == 'rotaviral_entiritis') == len(only_men), err_msg


def test_disability_weights():
    simulation = make_simulation_object()
    disability_weight = simulation.values.get_value('disability_weight')

    pump_simulation(simulation, iterations=1)

    time_step = config.simulation_parameters.time_step
    mild_disability_weight = get_disability_weight(
        healthstate_id=causes.mild_diarrhea.disability_weight) * time_step / 365
    moderate_disability_weight = get_disability_weight(
        healthstate_id=causes.moderate_diarrhea.disability_weight) * time_step / 365
    severe_disability_weight = get_disability_weight(
        healthstate_id=causes.severe_diarrhea.disability_weight)*time_step/365

    only_men = simulation.population.population.query("sex == 'Male'")
    simulation_weights = np.sort(pd.unique(disability_weight(only_men.index)))
    GBD_weights = np.sort([mild_disability_weight, moderate_disability_weight, severe_disability_weight])

    assert np.allclose(simulation_weights, GBD_weights), "Unexpected disability weights values."

    mild_diarrhea_index = simulation.population.population.query("diarrhea == 'mild_diarrhea'").index
    moderate_diarrhea_index = simulation.population.population.query("diarrhea == 'moderate_diarrhea'").index
    severe_diarrhea_index = simulation.population.population.query("diarrhea == 'severe_diarrhea'").index

    err_msg = "diarrhea severity state should be correctly mapped to its specific disability weight"
    assert np.allclose(pd.unique(disability_weight(mild_diarrhea_index)), mild_disability_weight), err_msg
    assert np.allclose(pd.unique(disability_weight(moderate_diarrhea_index)), moderate_disability_weight), err_msg
    assert np.allclose(pd.unique(disability_weight(severe_diarrhea_index)), severe_disability_weight), err_msg


def test_remission():
    diarrhea_and_etiology_models = build_diarrhea_model()
    simulation = setup_simulation([generate_test_population] + diarrhea_and_etiology_models)

    # Make it so that duration of diarrhea is 1 day among all men
    # except for men over age 20, for whom duration will be 2 days.
    duration_data = build_table(1, ['age', 'year', 'sex', 'duration'])
    duration_data.loc[duration_data.age >= 20, 'duration'] = 2
    duration = simulation.values.get_value('dwell_time.mild_diarrhea')
    duration.source = simulation.tables.build_table(duration_data)
    duration = simulation.values.get_value('dwell_time.moderate_diarrhea')
    duration.source = simulation.tables.build_table(duration_data)
    duration = simulation.values.get_value('dwell_time.severe_diarrhea')
    duration.source = simulation.tables.build_table(duration_data)

    # make it so that some men will get diarrhea due to rota
    incidence = build_table(0)
    incidence.loc[incidence.sex == 'Male', 'rate'] = 200
    rota_incidence = simulation.values.get_rate('incidence_rate.rotaviral_entiritis')
    rota_incidence.source = simulation.tables.build_table(incidence)

    pump_simulation(simulation, iterations=1)

    pop = simulation.population.population
    diarrhea_first_time_step = pop[pop.diarrhea_event_time.notnull()]
    # check that everyone has been moved into the diarrhea state
    err_msg = "duration should correctly determine the duration of a bout of diarrhea"
    assert (set(pd.unique(diarrhea_first_time_step.diarrhea))
            == {'mild_diarrhea', 'moderate_diarrhea', 'severe_diarrhea'}), err_msg

    pump_simulation(simulation, iterations=1)


    # Make sure that at least some male simulants under 20 remitted into
    # the healthy state (some may not be healthy if they got diarrhea again).
    had_diarrhea = simulation.population.population.loc[diarrhea_first_time_step.index]
    err_msg = "Diarrhea duration incorrectly calculated."
    assert "healthy" in pd.unique(had_diarrhea.query("age < 20 and sex == 'Male'").diarrhea), err_msg

    # Make sure that all simulants between the over age 20 that got
    # diarrhea in the first time step still have diarrhea.
    assert "healthy" not in pd.unique(had_diarrhea.query("age >= 20 and sex == 'Male'").diarrhea), err_msg


# Test that severe_diarrhea is the only severity level of diarrhea associated with an excess mortality.
def test_diarrhea_elevated_mortality():
    diarrhea_and_etiology_models = build_diarrhea_model()
    simulation = setup_simulation([generate_test_population, Mortality()] + diarrhea_and_etiology_models)

    incidence = build_table(0)
    incidence.loc[incidence.sex == 'Male', 'rate'] = 14000
    rota_incidence = simulation.values.get_rate('incidence_rate.rotaviral_entiritis')
    rota_incidence.source = simulation.tables.build_table(incidence)

    # make the base mortality_rate 0

    mortality_rate = simulation.values.get_rate('mortality_rate')
    mortality_rate.source = simulation.tables.build_table(build_table(0))
    pump_simulation(simulation, iterations=1)

    pop = simulation.population.population
    mild_diarrhea_index = pop.query("diarrhea == 'mild_diarrhea'").index
    moderate_diarrhea_index = pop.query("diarrhea == 'moderate_diarrhea'").index
    severe_diarrhea_index = pop.query("diarrhea == 'severe_diarrhea'").index

    total_mortality = mortality_rate(pop.index).sum(axis=1)
    err_msg = "People with severe diarrhea should have an elevated mortality rate."
    assert total_mortality[severe_diarrhea_index].all(), err_msg

    err_msg = "People with mild/moderate diarrhea should have no elevated mortality."
    assert not total_mortality[mild_diarrhea_index].any(), err_msg
    assert not total_mortality[moderate_diarrhea_index].any(), err_msg


def test_severity_proportions():
    diarrhea_and_etiology_models = build_diarrhea_model()
    simulation = setup_simulation([generate_test_population] + diarrhea_and_etiology_models,
                                  population_size=10000)
    # Give everyone diarrhea
    incidence = build_table(14000)
    rota_incidence = simulation.values.get_rate('incidence_rate.rotaviral_entiritis')
    rota_incidence.source = simulation.tables.build_table(incidence)

    pump_simulation(simulation, iterations=1)
    pop = simulation.population.population

    mild_proportion_in_sim = len(pop.query("diarrhea == 'mild_diarrhea'"))/10000
    moderate_proportion_in_sim = len(pop.query("diarrhea == 'moderate_diarrhea'"))/10000
    severe_proportion_in_sim = len(pop.query("diarrhea == 'severe_diarrhea'"))/10000

    mild_proportion_in_GBD = get_severity_splits(causes.diarrhea.incidence, causes.mild_diarrhea.incidence)
    moderate_proportion_in_GBD = get_severity_splits(causes.diarrhea.incidence, causes.moderate_diarrhea.incidence)
    severe_proportion_in_GBD = get_severity_splits(causes.diarrhea.incidence, causes.severe_diarrhea.incidence)

    err_msg = "Severity splits not calculated correctly."
    assert np.isclose(mild_proportion_in_sim, mild_proportion_in_GBD, atol=.01), err_msg
    assert np.isclose(moderate_proportion_in_sim, moderate_proportion_in_GBD, atol=.01), err_msg
    assert np.isclose(severe_proportion_in_sim, severe_proportion_in_GBD, atol=.01), err_msg


# Test that diarrhea csmr is deleted from the background mortality rate.
def test_cause_deletion():
    config.simulation_parameters.set_with_metadata('initial_age', 0, layer='override', source=os.path.realpath(__file__))
    diarrhea_and_etiology_models = build_diarrhea_model()
    simulation = setup_simulation([generate_test_population, Mortality()] + diarrhea_and_etiology_models)

    cause_deleted_mr = get_cause_deleted_mortality_rate([get_cause_specific_mortality(causes.diarrhea.mortality)])
    simulation_mortality_rate = simulation.values.get_rate('mortality_rate')

    # compare for the earliest age group (this test requires
    # that generate_test_population is set to create a cohort of newborns)
    cause_deleted_mr_values = cause_deleted_mr.query(
        "year==2005 and age<.01").cause_deleted_mortality_rate.values

    simulation_values = simulation_mortality_rate(
        simulation.population.population.index).death_due_to_other_causes.unique()
    time_step = config.simulation_parameters.time_step

    err_msg = "Diarrhea has been deleted from the background mortality rate."
    assert np.allclose(np.sort(simulation_values),
                       np.sort(cause_deleted_mr_values * time_step/365), atol=.000001), err_msg


