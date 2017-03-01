from ceam_public_health.components.diarrhea_disease_model import diarrhea_factory
from ceam_public_health.components.interventions.rota_vaccine import determine_who_should_receive_dose
from ceam_tests.util import pump_simulation, generate_test_population, setup_simulation

def test_determine_who_should_receive_dose():
    """ 
    Determine if people are receiving the correct dosages. Move the simulation forward a few times to make sure that people who should get the vaccine do get the vaccine
    """
    factory = diarrhea_factory()

    simulation = setup_simulation([generate_test_population] + factory)

    pop = simulation.population.population

    pop['age'] = 0

    pop['rotaviral_entiritis_vaccine_first_dose'] = 0

    pump_simulation(simulation, time_step_days=365, iterations=1)

    first_dose_pop = determine_who_should_receive_dose(pop, pop.index, 'rotaviral_entiritis_vaccine_first_dose', 1)
   
    import pdb; pdb.set_trace() 

def test_incidence_rate():
    """
    Set vaccine working column for only some people, ensure that their diarrhea due to rota incidence is reduced by the vaccine_effectiveness specified in the config file
    """
