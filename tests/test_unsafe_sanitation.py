import pandas as pd
import numpy as np
from ceam_public_health.components.risks.unsafe_sanitation import UnsafeSanitation
from ceam_tests.util import build_table, setup_simulation, generate_test_population, pump_simulation
from ceam_inputs import get_relative_risks, get_etiology_specific_incidence, get_pafs, get_exposures
from ceam_public_health.components.diarrhea_disease_model import diarrhea_factory
from datetime import datetime

def test_unsafe_sanitation():
    simulation = setup_simulation(components=[generate_test_population, UnsafeSanitation()] + diarrhea_factory(), start=datetime(2005, 1, 1))

    exposure = build_table(0, ['age', 'year', 'sex', 'cat1'])
    exposure['cat2'] = 0
    exposure['cat3'] = 1

    exp = simulation.values.get_rate('unsafe_sanitation.exposure')
    exp.source = simulation.tables.build_table(exposure)

    pump_simulation(simulation, time_step_days=1, iterations=1, year_start=2005)

    rota_inc = simulation.values.get_rate('incidence_rate.diarrhea_due_to_rotaviral_entiritis')
    rota_inc_unexposed = rota_inc(simulation.population.population.index)

    exposure = build_table(1, ['age', 'year', 'sex', 'cat1'])
    exposure['cat2'] = 0
    exposure['cat3'] = 0

    exp.source = simulation.tables.build_table(exposure)

    pump_simulation(simulation, time_step_days=365, iterations=1, year_start=2005)

    # rota_inc = simulation.values.get_rate('incidence_rate.diarrhea_due_to_rotaviral_entiritis')
    rota_inc_exposed = rota_inc(simulation.population.population.index)

    assert np.all(rota_inc_exposed > rota_inc_unexposed), "incidence rate should be higher among exposed simulants"
