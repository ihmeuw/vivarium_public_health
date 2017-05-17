import pandas as pd
import numpy as np
import pytest
from datetime import timedelta, datetime

from ceam import config
from ceam.framework.event import Event
from ceam_tests.util import (pump_simulation, generate_test_population,
                             setup_simulation, build_table)

from ceam_public_health.components.diarrhea_disease_model import diarrhea_factory
from ceam_public_health.components.interventions.rota_vaccine import (set_vaccine_duration,
                                                                      determine_vaccine_protection,
                                                                      wane_immunity)
from ceam_public_health.components.interventions.rota_vaccine import RotaVaccine
from ceam_public_health.components.base_population import (generate_base_population,
                                                           age_simulants)


# 1: determine_who_should_receive_dose
def test_determine_who_should_receive_dose():
    """
    Determine if people are receiving the correct dosages.
    ove the simulation forward a few times to make sure that people who should
    get the vaccine do get the vaccine
    """
    factory = diarrhea_factory()

    rv_instance = RotaVaccine(True)

    simulation = setup_simulation([generate_test_population, rv_instance] + \
                                  factory, population_size=10000)

    pop = simulation.population.population

    pop['rotaviral_entiritis_vaccine_first_dose'] = 0

    pop['fractional_age'] = config.rota_vaccine.age_at_first_dose / 365

    first_dose_pop = rv_instance.determine_who_should_receive_dose(pop,
                                                                   'rotaviral_entiritis_vaccine',
                                                                   1)

    # FIXME: This test will fail in years in which there is vaccination
    #     coverage in the baseline scenario
    assert np.allclose(len(pop)*config.rota_vaccine.vaccination_proportion_increase,
           len(first_dose_pop), .1), "determine who should receive dose needs" + \
                                     "to give doses at the correct age"

    first_dose_pop['rotaviral_entiritis_vaccine_second_dose'] = 0

    first_dose_pop['fractional_age'] = config.rota_vaccine.age_at_second_dose / 365

    second_dose_pop = rv_instance.determine_who_should_receive_dose(first_dose_pop,
                                                                    'rotaviral_entiritis_vaccine',
                                                                    2)

    # FIXME: This test will fail in years in which there is vaccination
    #     coverage in the baseline scenario
    assert np.allclose(len(pop) * config.rota_vaccine.vaccination_proportion_increase * \
        config.rota_vaccine.second_dose_retention,  len(second_dose_pop), .1), \
        "determine who should receive dose needs to give doses at the correct age"

    second_dose_pop['rotaviral_entiritis_vaccine_third_dose'] = 0

    second_dose_pop['fractional_age'] = config.rota_vaccine.age_at_third_dose / 365

    third_dose_pop = rv_instance.determine_who_should_receive_dose(second_dose_pop,
                                                                   'rotaviral_entiritis_vaccine',
                                                                   3)

    # FIXME: This test will fail in years in which there is vaccination
    #     coverage in the baseline scenario
    assert np.allclose(len(pop) * config.rota_vaccine.vaccination_proportion_increase * \
        config.rota_vaccine.second_dose_retention*config.rota_vaccine.third_dose_retention, \
        len(third_dose_pop), .1), "determine who should receive dose needs to give doses at the correct age"


# 2: set_vaccine_duration
def test_set_vaccine_duration():
    simulation = setup_simulation([generate_test_population])
    pop = simulation.population.population
    pop['rotaviral_entiritis_vaccine_first_dose_event_time'] = simulation.current_time
    new_pop = set_vaccine_duration(pop, "rotaviral_entiritis", "first")

    time_after_dose_at_which_immunity_is_conferred = pd.to_timedelta(1, unit='D')
    vaccine_full_immunity_duration = pd.to_timedelta(config.rota_vaccine.vaccine_full_immunity_duration, unit='D')
    waning_immunity_time = pd.to_timedelta(config.rota_vaccine.waning_immunity_time, unit='D')

    # assert that the vaccine starts two weeks after its administered
    assert np.all(new_pop['rotaviral_entiritis_vaccine_first_dose_immunity_start_time'] == simulation.current_time + \
                  time_after_dose_at_which_immunity_is_conferred)

    # # assert that the vaccine ends 2 years after it starts to have an effect
    assert np.all(new_pop['rotaviral_entiritis_vaccine_first_dose_immunity_end_time'] == simulation.current_time + \
                  time_after_dose_at_which_immunity_is_conferred + \
                  vaccine_full_immunity_duration + waning_immunity_time)


@pytest.fixture
def get_indexes():
    simulation = setup_simulation([generate_test_population, age_simulants,
                                   RotaVaccine(True)])

    pump_simulation(simulation, iterations=8)

    not_vaccinated = simulation.population.population.query(
        "rotaviral_entiritis_vaccine_first_dose_is_working == 0")

    vaccinated = simulation.population.population.query(
        "rotaviral_entiritis_vaccine_first_dose_is_working == 1")

    return vaccinated, not_vaccinated


# 3: set_working_column
def test_set_working_column1(get_indexes):

    vaccinated, not_vaccinated = get_indexes

    # find an example of simulants of the same age and sex, but not vaccination
    #     status, and then compare their incidence rates
    assert np.allclose(len(vaccinated)/100, config.rota_vaccine.vaccination_proportion_increase,
        atol=.1), "working column should ensure that the correct number of simulants are receiving the benefits of the vaccine"

    assert np.all(vaccinated["rotaviral_entiritis_vaccine_first_dose_is_working"] == 1), "everyone that receives vaccine should have the working col set to true here"


def test_set_working_column2(get_indexes):
    # pump the simulation far enough ahead that simulants can get second dose
    simulation = setup_simulation([generate_test_population, age_simulants,
                                   RotaVaccine(True)] + diarrhea_factory())

    pump_simulation(simulation, iterations=14)

    vaccinated, not_vaccinated = get_indexes

    vaccinated = simulation.population.population.loc[vaccinated.index]
    not_vaccinated = simulation.population.population.loc[not_vaccinated.index]

    assert np.all(vaccinated["rotaviral_entiritis_vaccine_second_dose_is_working"] == 1)
    assert np.all(vaccinated["rotaviral_entiritis_vaccine_first_dose_is_working"] == 0)
    assert np.all(not_vaccinated["rotaviral_entiritis_vaccine_second_dose_is_working"] == 0)
    assert np.all(not_vaccinated["rotaviral_entiritis_vaccine_first_dose_is_working"] == 0)


def test_set_working_column3(get_indexes):
    # 19 days in, we should see that the third vaccine is working and the first
    #     and second are not
    simulation = setup_simulation([generate_test_population, age_simulants,
                                   RotaVaccine(True)] + diarrhea_factory())

    pump_simulation(simulation, iterations=20)

    vaccinated, not_vaccinated = get_indexes

    vaccinated = simulation.population.population.loc[vaccinated.index]
    not_vaccinated = simulation.population.population.loc[not_vaccinated.index]

    assert np.all(vaccinated["rotaviral_entiritis_vaccine_third_dose_is_working"] == 1)
    assert np.all(vaccinated["rotaviral_entiritis_vaccine_first_dose_is_working"] == 0)
    assert np.all(vaccinated["rotaviral_entiritis_vaccine_second_dose_is_working"] == 0)
    assert np.all(not_vaccinated["rotaviral_entiritis_vaccine_third_dose_is_working"] == 0)


def test_set_working_column4():
    # 39 days in, we should see that none of the vaccines are working
    simulation = setup_simulation([generate_test_population, age_simulants,
                                   RotaVaccine(True)] + diarrhea_factory())

    pump_simulation(simulation, iterations=61)

    pop = simulation.population.population

    assert np.all(pop["rotaviral_entiritis_vaccine_third_dose_is_working"] == 0)
    assert np.all(pop["rotaviral_entiritis_vaccine_first_dose_is_working"] == 0)
    assert np.all(pop["rotaviral_entiritis_vaccine_second_dose_is_working"] == 0)


# 4: incidence_rates
def test_incidence_rates():

    simulation = setup_simulation([generate_test_population, age_simulants,
                                   RotaVaccine(True)] + diarrhea_factory())

    rota_table = build_table(7000, ['age', 'year', 'sex', 'cat1'])

    rota_inc = simulation.values.get_rate('incidence_rate.rotaviral_entiritis')

    rota_inc.source = simulation.tables.build_table(
        rota_table)

    vaccine_effectiveness = config.rota_vaccine.first_dose_effectiveness

    # pump the simulation far enough ahead that simulants can get first dose
    pump_simulation(simulation, duration=timedelta(days=7))

    not_vaccinated = simulation.population.population.query(
                    "rotaviral_entiritis_vaccine_first_dose_is_working == 0")

    vaccinated = simulation.population.population.query(
                    "rotaviral_entiritis_vaccine_first_dose_is_working == 1")

    # find an example of simulants of the same age and sex, but not vaccination
    #     status, and then compare their incidence rates
    assert np.allclose(pd.unique(rota_inc(vaccinated.index)), pd.unique(
        rota_inc(not_vaccinated.index)*(1-vaccine_effectiveness))), \
        "simulants that receive vaccine should have lower incidence of diarrhea" \
        "due to rota"

    # now try with two doses
    simulation = setup_simulation([generate_test_population, age_simulants,
                                   RotaVaccine(True)] + diarrhea_factory())

    rota_table = build_table(1000, ['age', 'year', 'sex', 'cat1'])

    rota_inc = simulation.values.get_rate('incidence_rate.rotaviral_entiritis')

    rota_inc.source = simulation.tables.build_table(
        rota_table)

    vaccine_effectiveness = config.rota_vaccine.second_dose_effectiveness

    # pump the simulation far enough ahead that simulants can get second dose
    pump_simulation(simulation, duration=timedelta(days=13))

    not_vaccinated = simulation.population.population.query(
        "rotaviral_entiritis_vaccine_second_dose_is_working == 0")

    vaccinated = simulation.population.population.query(
        "rotaviral_entiritis_vaccine_second_dose_is_working == 1")

    # find an example of simulants of the same age and sex, but not vaccination
    #     status, and then compare their incidence rates
    assert np.allclose(pd.unique(rota_inc(vaccinated.index)), pd.unique(
        rota_inc(not_vaccinated.index)*(1-vaccine_effectiveness))), \
        "simulants that receive vaccine should have lower incidence of diarrhea due to rota"

    # now try with three doses
    simulation = setup_simulation([generate_test_population, age_simulants,
                                   RotaVaccine(True)] + diarrhea_factory())

    rota_table = build_table(1000, ['age', 'year', 'sex', 'cat1'])

    rota_inc = simulation.values.get_rate('incidence_rate.rotaviral_entiritis')

    rota_inc.source = simulation.tables.build_table(
        rota_table)

    vaccine_effectiveness = config.rota_vaccine.third_dose_effectiveness

    # pump the simulation far enough ahead that simulants can get third dose
    pump_simulation(simulation, duration=timedelta(days=19))

    not_vaccinated = simulation.population.population.query(
                     "rotaviral_entiritis_vaccine_third_dose_is_working == 0")

    vaccinated = simulation.population.population.query(
                     "rotaviral_entiritis_vaccine_third_dose_is_working == 1")

    # find an example of simulants of the same age and sex, but not vaccination
    #     status, and then compare their incidence rates
    assert np.allclose(pd.unique(rota_inc(vaccinated.index)), pd.unique(
        rota_inc(not_vaccinated.index)*(1-vaccine_effectiveness))), \
        "simulants that receive vaccine should have lower incidence of diarrhea due to rota"


# 5. determine_vaccine_protection
def test_determine_vaccine_protection():
    simulation = setup_simulation([generate_test_population, age_simulants, RotaVaccine(True)])

    # pump the simulation forward
    pump_simulation(simulation, duration=timedelta(days=8))

    population = simulation.population.population
    dose_working_index = population.query(
        "rotaviral_entiritis_vaccine_first_dose_is_working == 1").index

    duration = config.rota_vaccine.vaccine_full_immunity_duration
    effectiveness = config.rota_vaccine.first_dose_effectiveness
    waning_immunity_time = config.rota_vaccine.waning_immunity_time

    series = determine_vaccine_protection(population, dose_working_index,
                                          wane_immunity, simulation.current_time,
                                          "first", effectiveness)

    assert np.allclose(series, effectiveness), "determine vaccine effectiveness" + \
                                     " should return the correct effectiveness" + \
                                     " for each simulant based on vaccination status" + \
                                     "and time since vaccination"

    assert len(series) == len(dose_working_index), "number of effectiveness estimates that are" + \
                                                   " returned matches the number" + \
                                                   " of simulants who have their working"


def test_wane_immunity():
    assert np.allclose(.25, wane_immunity(30, 20, 20, .5)), \
        "vaccine should confer 50% as much benefit when halfway through" + \
        "the waning period"
    assert np.allclose(0, wane_immunity(41, 20, 20, .5)), "vaccine should" + \
        "confer no benefit when after the waning period is over"


def test_rota_vaccine_coverage():
    # create a simulation object where there is no intervention
    config.simulation_parameters.year_start = 2014

    simulation = setup_simulation([generate_test_population, age_simulants,
                                   RotaVaccine(False)], population_size=10000)

    # pump the simulation forward
    pump_simulation(simulation, duration=timedelta(days=8))

    assert len(simulation.population.population.rotaviral_entiritis_vaccine_first_dose_is_working) > 0, \
        "ensure that, even when there is no intervention, there is still some baseline coverage that exists"

    # now create a simulation where we are including an intevention
    simulation2 = setup_simulation([generate_test_population, age_simulants,
                                    RotaVaccine(True)], population_size=10000)

    # pump the simulation forward
    pump_simulation(simulation2, duration=timedelta(days=8))

    assert np.allclose(len(simulation.population.population.query(
        "rotaviral_entiritis_vaccine_first_dose_is_working == 1"))/10000 + \
        config.rota_vaccine.vaccination_proportion_increase,
        len(simulation2.population.population.query(
            "rotaviral_entiritis_vaccine_first_dose_is_working == 1"))/10000,
        rtol=.05), "when including an intervention, ensure that the" + \
                   "intervention increases the proportion of people that get" + \
                   "vaccinated"
