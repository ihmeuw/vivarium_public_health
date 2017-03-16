import pytest

from datetime import timedelta

import numpy as np

from ceam.framework.state_machine import Transition, State

from ceam_tests.util import setup_simulation, pump_simulation, build_table, generate_test_population

from ceam_public_health.components.metrics import Metrics
from ceam_public_health.components.base_population import generate_base_population
from ceam_public_health.components.disease import DiseaseState, RateTransition, ExcessMortalityState, DiseaseModel

def test_years_lived_with_disability():
    model = DiseaseModel('state')
    healthy_state = ExcessMortalityState('healthy', disability_weight=0.2, excess_mortality_data=build_table(0), prevalence_data=build_table(0, ['age', 'year', 'sex', 'prevalence']), csmr_data=build_table(0)) # In a world where even healthy people have disability weight...
    unreachable_state = ExcessMortalityState('sick', disability_weight=0.4, excess_mortality_data=build_table(0), prevalence_data=build_table(0, ['age', 'year', 'sex', 'prevalence']), csmr_data=build_table(0)) # No one can get into this state so it should never contribute weight to the population total
    model.states.append(healthy_state)

    simulation = setup_simulation([generate_base_population, model, Metrics()], population_size=1000)

    metrics = simulation.values.get_value('metrics')

    assert metrics(simulation.population.population.index)['years_lived_with_disability'] == 0

    pump_simulation(simulation, duration=timedelta(days=365))

    assert np.isclose(metrics(simulation.population.population.index)['years_lived_with_disability'], 1000 * 0.2, rtol=0.01)
