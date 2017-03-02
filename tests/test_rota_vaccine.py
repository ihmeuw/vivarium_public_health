from ceam_public_health.components.diarrhea_disease_model import diarrhea_factory
from ceam_public_health.components.interventions.rota_vaccine import determine_who_should_receive_dose
from ceam_tests.util import pump_simulation, generate_test_population, setup_simulation
from ceam import config
from ceam.framework.event import Event
from ceam_inputs import generate_ceam_population
import numpy as np

def test_determine_who_should_receive_dose():
    """ 
    Determine if people are receiving the correct dosages. Move the simulation forward a few times to make sure that people who should get the vaccine do get the vaccine
    """
    factory = diarrhea_factory()

    simulation = setup_simulation([generate_test_population] + factory)
    emitter = simulation.events.get_emitter('time_step')

    pop = simulation.population.population

    pop['rotaviral_entiritis_vaccine_first_dose'] = 0

    pop['age'] = config.getint('rota_vaccine', 'age_at_first_dose') / 365

    first_dose_pop = determine_who_should_receive_dose(pop, pop.index, 'rotaviral_entiritis_vaccine_first_dose', 1)

    # FIXME: This test will fail in years in which there is vaccination coverage in the baseline scenario
    assert np.allclose(len(pop)*config.getfloat('rota_vaccine', 'vaccination_proportion_increase'),  len(first_dose_pop), .1), "determine who should receive dose needs to give doses at the correct age"

    first_dose_pop['rotaviral_entiritis_vaccine_second_dose'] = 0

    first_dose_pop['age'] = (config.getint('rota_vaccine', 'age_at_first_dose') + 61) / 365

    second_dose_pop = determine_who_should_receive_dose(first_dose_pop, first_dose_pop.index, 'rotaviral_entiritis_vaccine_second_dose', 2)

    # FIXME: This test will fail in years in which there is vaccination coverage in the baseline scenario
    assert np.allclose(len(pop)*config.getfloat('rota_vaccine', 'vaccination_proportion_increase')*config.getfloat('rota_vaccine', 'second_dose_retention'),  len(second_dose_pop), .1), "determine who should receive dose needs to give doses at the correct age"

    second_dose_pop['rotaviral_entiritis_vaccine_third_dose'] = 0

    second_dose_pop['age'] = (config.getint('rota_vaccine', 'age_at_first_dose') + 61 + 61) / 365

    third_dose_pop = determine_who_should_receive_dose(second_dose_pop, second_dose_pop.index, 'rotaviral_entiritis_vaccine_third_dose', 3)

    # FIXME: This test will fail in years in which there is vaccination coverage in the baseline scenario
    assert np.allclose(len(pop)*config.getfloat('rota_vaccine', 'vaccination_proportion_increase')*config.getfloat('rota_vaccine', 'second_dose_retention')*config.getfloat('rota_vaccine', 'third_dose_retention'),  len(third_dose_pop), .1), "determine who should receive dose needs to give doses at the correct age"


def test_incidence_rate():
    """
    Set vaccine working column for only some people, ensure that their diarrhea due to rota incidence is reduced by the vaccine_effectiveness specified in the config file
    """
