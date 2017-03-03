import pandas as pd
import numpy as np
from ceam_public_health.components.interventions.ors_supplementation import ORS
from ceam_public_health.components.diarrhea_disease_model import diarrhea_factory
from ceam_tests.util import pump_simulation, generate_test_population, setup_simulation, build_table
from ceam import config
from ceam_inputs import get_ors_exposure
from ceam.framework.event import Event
from ceam_public_health.components.base_population import generate_base_population
import pytest

def test_ors_exposure_effect():
    """
    Checks that the ors_clock is working and that people that receive ors see a decrease in mortality
    """
    factory = diarrhea_factory()

    # FIXME: This function will only work if all simulants are the same age
    simulation = setup_simulation([generate_base_population, ORS()] + factory)
    emitter = simulation.events.get_emitter('time_step')

    simulation.population.population.age = 0
    simulation.population.population.fractional_age = 0

    # make the incidence really high
    inc = build_table(1000)

    rota_inc = simulation.values.get_rate('incidence_rate.diarrhea_due_to_rotaviral_entiritis')

    rota_inc.source = simulation.tables.build_table(inc)

    # make ors exposure 50/50 so that we have some people that get ors, some that don't. will compare later to make sure that people that receive ORS have decreased mortality
    ors = get_ors_exposure()

    ors['cat1'] = .5

    ors['cat2'] = .5

    ors_exposure = simulation.values.get_rate('exposure.ors')

    ors_exposure.source = simulation.tables.build_table(ors)

    # get mortality rate before ORS is applied in time step
    mortality_rate = simulation.values.get_value('excess_mortality.diarrhea')

    # pump the simulation forward 5 time period
    # pump_simulation(simulation, time_step_days=5, iterations=1)
    emitter(Event(simulation.population.population.index))

    # build test pops
    male_ors_pop = simulation.population.population.query("ors_clock == 1 and sex == 'Male'")
    male_no_ors_pop = simulation.population.population.query("ors_clock == 0 and sex == 'Male'")

    mortality_rate_ors = mortality_rate(male_ors_pop.index)
    mortality_rate_no_ors = mortality_rate(male_no_ors_pop.index)

    assert pd.unique(mortality_rate_no_ors) == pd.unique(mortality_rate_ors) * config.getfloat('ORS', 'ors_effectiveness'), "people that receive ORS should have a decreased excess mortality"


# End.
