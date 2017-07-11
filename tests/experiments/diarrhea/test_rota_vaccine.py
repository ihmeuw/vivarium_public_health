import os
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from vivarium import config
from vivarium.test_util import (pump_simulation, generate_test_population,
                             setup_simulation, build_table)

from ceam_public_health.population import age_simulants

from ceam_public_health.experiments.diarrhea.components.diarrhea import diarrhea_factory
from ceam_public_health.experiments.diarrhea.components.rota_vaccine import RotaVaccine
from ceam_public_health.experiments.diarrhea.components.rota_vaccine import (set_vaccine_duration,
                                                                             determine_vaccine_protection,
                                                                             wane_immunity)

from ceam_inputs import get_rota_vaccine_rrs


def setup():
    try:
        config.reset_layer('override', preserve_keys=['input_data.intermediary_data_cache_path',
                                                      'input_data.auxiliary_data_folder'])
    except KeyError:
        pass

    config.simulation_parameters.set_with_metadata('year_start', 2010, layer='override',
                                                   source=os.path.realpath(__file__))
    config.simulation_parameters.set_with_metadata('year_end', 2015, layer='override',
                                                   source=os.path.realpath(__file__))
    config.simulation_parameters.set_with_metadata('time_step', 1, layer='override',
                                                   source=os.path.realpath(__file__))
    config.simulation_parameters.set_with_metadata('initial_age', 0, layer='override',
                                                   source=os.path.realpath(__file__))
    config.simulation_parameters.set_with_metadata('location_id', 179, layer='override',
                                                   source=os.path.realpath(__file__))
    config.rota_vaccine.set_with_metadata('age_at_first_dose', 6, layer='override', source=os.path.realpath(__file__))
    config.rota_vaccine.set_with_metadata('age_at_second_dose', 12, layer='override', source=os.path.realpath(__file__))
    config.rota_vaccine.set_with_metadata('age_at_third_dose', 18, layer='override', source=os.path.realpath(__file__))
    config.rota_vaccine.set_with_metadata('time_after_dose_at_which_immunity_is_conferred', 1,
                                          layer='override', source=os.path.realpath(__file__))
    config.rota_vaccine.set_with_metadata('vaccine_full_immunity_duration', 20,
                                          layer='override', source=os.path.realpath(__file__))
    config.rota_vaccine.set_with_metadata('waning_immunity_time', 20, layer='override',
                                          source=os.path.realpath(__file__))
    config.rota_vaccine.set_with_metadata('vaccination_proportion_increase', 0.1, layer='override',
                                          source=os.path.realpath(__file__))
    config.rota_vaccine.set_with_metadata('second_dose_retention', 1, layer='override',
                                          source=os.path.realpath(__file__))
    config.rota_vaccine.set_with_metadata('third_dose_retention', 1, layer='override',
                                          source=os.path.realpath(__file__))


def test_determine_who_should_receive_dose():
    """
    Determine if people are receiving the correct dosages.
    Move the simulation forward a few times to make sure that people who should
    get the vaccine do get the vaccine
    """
    config.rota_vaccine.set_with_metadata('second_dose_retention', 0.8, layer='override',
                                          source=os.path.realpath(__file__))
    config.rota_vaccine.set_with_metadata('third_dose_retention', 0.4, layer='override',
                                          source=os.path.realpath(__file__))
    config.rota_vaccine.set_with_metadata('dtp3_coverage', 0, layer='override',
                                          source=os.path.realpath(__file__))

    rv_instance = RotaVaccine()
    simulation = setup_simulation([generate_test_population, rv_instance] + diarrhea_factory(), population_size=10000)

    pop = simulation.population.population
    pop['rotaviral_entiritis_vaccine_first_dose'] = 0
    pop['age'] = config.rota_vaccine.age_at_first_dose / 365

    first_dose_pop = rv_instance.determine_who_should_receive_dose(pop, 'rotaviral_entiritis_vaccine', 1, simulation.current_time)

    # FIXME: This test will fail in years in which there is vaccination coverage in the baseline scenario
    err_msg = ("Determine who should receive dose should give doses "
               "to the correct proportion of sims at the correct age.")
    assert np.isclose(len(pop)*config.rota_vaccine.vaccination_proportion_increase,
                      len(first_dose_pop), rtol=.1), err_msg

    first_dose_pop['rotaviral_entiritis_vaccine_second_dose'] = 0
    first_dose_pop['age'] = config.rota_vaccine.age_at_second_dose / 365
    second_dose_pop = rv_instance.determine_who_should_receive_dose(first_dose_pop, 'rotaviral_entiritis_vaccine', 2, simulation.current_time)

    # FIXME: This test will fail in years in which there is vaccination coverage in the baseline scenario
    assert np.isclose(len(pop) * config.rota_vaccine.vaccination_proportion_increase
                      * config.rota_vaccine.second_dose_retention,  len(second_dose_pop), rtol=.1), err_msg

    second_dose_pop['rotaviral_entiritis_vaccine_third_dose'] = 0
    second_dose_pop['age'] = config.rota_vaccine.age_at_third_dose / 365
    third_dose_pop = rv_instance.determine_who_should_receive_dose(second_dose_pop, 'rotaviral_entiritis_vaccine', 3, simulation.current_time)

    # FIXME: This test will fail in years in which there is vaccination coverage in the baseline scenario
    assert np.allclose(len(pop) * config.rota_vaccine.vaccination_proportion_increase
                       * config.rota_vaccine.second_dose_retention
                       * config.rota_vaccine.third_dose_retention, len(third_dose_pop), rtol=.1), err_msg


def test_set_vaccine_duration():
    simulation = setup_simulation([generate_test_population])
    pop = simulation.population.population
    pop['rotaviral_entiritis_vaccine_first_dose_event_time'] = simulation.current_time
    new_pop = set_vaccine_duration(pop, "rotaviral_entiritis", "first")

    time_after_dose_at_which_immunity_is_conferred = pd.to_timedelta(1, unit='D')
    vaccine_full_immunity_duration = pd.to_timedelta(config.rota_vaccine.vaccine_full_immunity_duration, unit='D')
    waning_immunity_time = pd.to_timedelta(config.rota_vaccine.waning_immunity_time, unit='D')

    # assert that the vaccine starts two weeks after its administered
    assert np.all(new_pop['rotaviral_entiritis_vaccine_first_dose_immunity_start_time']
                  == simulation.current_time + time_after_dose_at_which_immunity_is_conferred)

    # # assert that the vaccine ends 2 years after it starts to have an effect
    assert np.all(new_pop['rotaviral_entiritis_vaccine_first_dose_immunity_end_time']
                  == simulation.current_time + time_after_dose_at_which_immunity_is_conferred
                  + vaccine_full_immunity_duration + waning_immunity_time)


@pytest.fixture
def get_indexes():
    simulation = setup_simulation([generate_test_population, age_simulants, RotaVaccine()], population_size=10000)
    
    config.rota_vaccine.set_with_metadata('dtp3_coverage', 0, layer='override',
                                          source=os.path.realpath(__file__))

    pump_simulation(simulation, iterations=8)

    not_vaccinated = simulation.population.population.query("rotaviral_entiritis_vaccine_first_dose_is_working == 0")
    vaccinated = simulation.population.population.query("rotaviral_entiritis_vaccine_first_dose_is_working == 1")

    return vaccinated, not_vaccinated


def test_set_working_column1(get_indexes):
    vaccinated, not_vaccinated = get_indexes


    err_msg = ("Working column should ensure that the correct "
               "number of simulants are receiving the benefits of the vaccine.")
    assert np.allclose(len(vaccinated)/(len(vaccinated)+len(not_vaccinated)), config.rota_vaccine.vaccination_proportion_increase, rtol=.1), err_msg

    err_msg = ("set_working_column needs to correctly identify who has "
               "been vaccinated and whether the vaccine should be conferring any benefit.")
    assert vaccinated["rotaviral_entiritis_vaccine_first_dose_is_working"].all(), err_msg
    assert not not_vaccinated["rotaviral_entiritis_vaccine_first_dose_is_working"].any(), err_msg
    assert not vaccinated["rotaviral_entiritis_vaccine_second_dose_is_working"].any(), err_msg
    assert not not_vaccinated["rotaviral_entiritis_vaccine_second_dose_is_working"].any(), err_msg
    assert not vaccinated["rotaviral_entiritis_vaccine_third_dose_is_working"].any(), err_msg
    assert not not_vaccinated["rotaviral_entiritis_vaccine_third_dose_is_working"].any(), err_msg


def test_set_working_column2(get_indexes):
    config.rota_vaccine.set_with_metadata('second_dose_retention', 1, layer='override',
                                          source=os.path.realpath(__file__))
    config.rota_vaccine.set_with_metadata('third_dose_retention', 1, layer='override',
                                          source=os.path.realpath(__file__))

    # pump the simulation far enough ahead that simulants can get second dose
    simulation = setup_simulation([generate_test_population, age_simulants, RotaVaccine()] + diarrhea_factory())
    pump_simulation(simulation, iterations=14)

    vaccinated, not_vaccinated = get_indexes

    vaccinated = simulation.population.population.loc[vaccinated.index]
    not_vaccinated = simulation.population.population.loc[not_vaccinated.index]

    err_msg = ("set_working_column needs to correctly identify who has "
               "been vaccinated and whether the vaccine should be conferring any benefit.")
    assert not vaccinated["rotaviral_entiritis_vaccine_first_dose_is_working"].any(), err_msg
    assert not not_vaccinated["rotaviral_entiritis_vaccine_first_dose_is_working"].any(), err_msg
    assert vaccinated["rotaviral_entiritis_vaccine_second_dose_is_working"].all(), err_msg
    assert not not_vaccinated["rotaviral_entiritis_vaccine_second_dose_is_working"].any(), err_msg
    assert not vaccinated["rotaviral_entiritis_vaccine_third_dose_is_working"].any(), err_msg
    assert not not_vaccinated["rotaviral_entiritis_vaccine_third_dose_is_working"].any(), err_msg


def test_set_working_column3(get_indexes):
    config.rota_vaccine.set_with_metadata('second_dose_retention', 1, layer='override',
                                          source=os.path.realpath(__file__))
    config.rota_vaccine.set_with_metadata('third_dose_retention', 1, layer='override',
                                          source=os.path.realpath(__file__))

    # 19 days in, we should see that the third vaccine is working and the first and second are not
    simulation = setup_simulation([generate_test_population, age_simulants, RotaVaccine()] + diarrhea_factory())
    pump_simulation(simulation, iterations=20)

    vaccinated, not_vaccinated = get_indexes

    vaccinated = simulation.population.population.loc[vaccinated.index]
    not_vaccinated = simulation.population.population.loc[not_vaccinated.index]

    err_msg = ("set_working_column needs to correctly identify who has "
               "been vaccinated and whether the vaccine should be conferring any benefit.")
    assert not vaccinated["rotaviral_entiritis_vaccine_first_dose_is_working"].any(), err_msg
    assert not not_vaccinated["rotaviral_entiritis_vaccine_first_dose_is_working"].any(), err_msg
    assert not vaccinated["rotaviral_entiritis_vaccine_second_dose_is_working"].any(), err_msg
    assert not not_vaccinated["rotaviral_entiritis_vaccine_second_dose_is_working"].any(), err_msg
    assert vaccinated["rotaviral_entiritis_vaccine_third_dose_is_working"].all(), err_msg
    assert not not_vaccinated["rotaviral_entiritis_vaccine_third_dose_is_working"].any(), err_msg


def test_set_working_column4():
    # 60 days in, we should see that none of the vaccines are working
    simulation = setup_simulation([generate_test_population, age_simulants, RotaVaccine()] + diarrhea_factory())

    pump_simulation(simulation, iterations=61)

    pop = simulation.population.population

    assert not pop["rotaviral_entiritis_vaccine_third_dose_is_working"].any()
    assert not pop["rotaviral_entiritis_vaccine_first_dose_is_working"].any()
    assert not pop["rotaviral_entiritis_vaccine_second_dose_is_working"].any()


def test_incidence_rates():
    config.simulation_parameters.set_with_metadata('year_start', 2014, layer='override',
                                                   source=os.path.realpath(__file__))
    config.rota_vaccine.set_with_metadata('vaccination_proportion_increase', .4, layer='override',
                                          source=os.path.realpath(__file__))

    simulation = setup_simulation([generate_test_population, age_simulants, RotaVaccine()] + diarrhea_factory())

    rota_table = build_table(7000)
    rota_inc = simulation.values.get_rate('incidence_rate.rotaviral_entiritis')
    rota_inc.source = simulation.tables.build_table(rota_table)

    vaccine_protection = config.rota_vaccine.first_dose_protection

    # pump the simulation far enough ahead that simulants can get first dose
    pump_simulation(simulation, duration=timedelta(days=20))

    not_vaccinated = simulation.population.population.query("rotaviral_entiritis_vaccine_third_dose_is_working == 0")
    vaccinated = simulation.population.population.query("rotaviral_entiritis_vaccine_third_dose_is_working == 1")

    # find an example of simulants of the same age and sex, but not vaccination
    # status, and then compare their incidence rates
    err_msg = "simulants that receive vaccine should have lower incidence of diarrhea due to rota"
    a = [pd.unique(rota_inc(vaccinated.index))]
    a.append(pd.unique(rota_inc(not_vaccinated.index)*(1-vaccine_protection)))
    rr = get_rota_vaccine_rrs()
    cov = simulation.values.get_value('rotaviral_entiritis_vaccine_coverage')
    cov = pd.unique(cov(vaccinated.index))
    paf = ((1 - cov) * (rr - 1)) / ((1 - cov) * (rr - 1) + 1)
    assert np.allclose(pd.unique(rota_inc(vaccinated.index) * rr),
                       pd.unique(rota_inc(not_vaccinated.index))), err_msg


def test_determine_vaccine_protection():
    simulation = setup_simulation([generate_test_population, age_simulants, RotaVaccine()])
    pump_simulation(simulation, duration=timedelta(days=8))
    population = simulation.population.population

    dose_working_index = population.query("rotaviral_entiritis_vaccine_first_dose_is_working == 1").index
    protection = config.rota_vaccine.first_dose_protection

    series = determine_vaccine_protection(population, dose_working_index, wane_immunity,
                                          simulation.current_time, "first", protection)

    assert np.allclose(series, protection), ("Determine vaccine protection should return the correct protection for"
                                             + " each simulant based on vaccination status and time since vaccination")
    assert len(series) == len(dose_working_index), ("Number of protection estimates that are returned "
                                                    + "matches the number of simulants who have their working")


def test_wane_immunity():
    assert np.allclose(.25, wane_immunity(30, 20, 20, .5)), ("Vaccine should confer 50% as much benefit "
                                                             + "when halfway through the waning period.")
    assert np.allclose(0, wane_immunity(41, 20, 20, .5)), ("Vaccine should confer no benefit "
                                                           + "after the waning period is over.")


def test_rota_vaccine_coverage():
    # create a simulation object where there is no intervention
    config.simulation_parameters.set_with_metadata('year_start', 2014, layer='override',
                                                   source=os.path.realpath(__file__))
    config.rota_vaccine.set_with_metadata('vaccination_proportion_increase', 0, layer='override',
                                          source=os.path.realpath(__file__))
    simulation = setup_simulation([generate_test_population, age_simulants, RotaVaccine()], population_size=10000)
    pump_simulation(simulation, duration=timedelta(days=8))

    err_msg = "Ensure that, even when there is no intervention, there is still some baseline coverage that exists."
    assert simulation.population.population.rotaviral_entiritis_vaccine_first_dose_is_working.any(), err_msg

    # now create a simulation where we are including an intevention
    config.rota_vaccine.set_with_metadata('vaccination_proportion_increase', .2, layer='override',
                                          source=os.path.realpath(__file__))
    simulation2 = setup_simulation([generate_test_population, age_simulants, RotaVaccine()], population_size=10000)
    pump_simulation(simulation2, duration=timedelta(days=8))

    base_vaccinated = simulation.population.population.query("rotaviral_entiritis_vaccine_first_dose_is_working == 1")
    intervention_vaccinated = simulation2.population.population.query(
        "rotaviral_entiritis_vaccine_first_dose_is_working == 1")
    err_msg = ("When including an intervention, ensure that the intervention "
               + "increases the proportion of people that get vaccinated.")
    assert np.isclose(len(base_vaccinated)/10000 + config.rota_vaccine.vaccination_proportion_increase,
                      len(intervention_vaccinated)/10000, rtol=.05), err_msg

