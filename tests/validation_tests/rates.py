import numpy as np
import pandas as pd

from vivarium.framework.state_machine import State
from vivarium.framework.utilities import from_yearly
from vivarium.testing_utilities import build_table, TestPopulation
from vivarium.interface.interactive import setup_simulation

from vivarium_public_health.disease import RateTransition, DiseaseModel


def make_model(config, incidence_rate, recovery_rate):
    year_start = config.time.start.year
    year_end = config.time.end.year
    healthy = State('healthy')
    sick = State('sick')
    recovered = State('recovered')

    healthy.transition_set.append(RateTransition(sick, 'infection',
                                                 build_table(incidence_rate, year_start, year_end)))
    sick.transition_set.append(RateTransition(healthy, 'recovery',
                                              build_table(recovery_rate, year_start, year_end)))

    model = DiseaseModel('simple_disease', initial_state=healthy)
    model.states.extend([healthy, sick, recovered])

    return model


def test_incidence_rate_recalculation(base_config):
    base_config.time.step_size = 1
    incidence_rate = 0.01
    recovery_rate = 72  # Average duration of 5 days
    sim = setup_simulation([TestPopulation(), make_model(base_config, incidence_rate, recovery_rate)], config)

    susceptible = [(sim.population.population.simple_disease == 'healthy').sum()]
    new_cases = []
    known_cases = pd.Index([])

    for step in range(360):
        sim.step()
        cases = sim.population.population.query('simple_disease == "sick"')
        new_cases.append(len(cases.index.difference(known_cases)))
        known_cases = cases.index
        susceptible.append((sim.population.population.simple_disease == 'healthy').sum())

    susceptible = susceptible[:-1]
    incidence_rates = np.array(new_cases)/np.array(susceptible)

    assert np.isclose(np.mean(incidence_rates), from_yearly(0.01, pd.Timedelta(days=1)), rtol=0.05)

    expected_prevalence = incidence_rate * (1/recovery_rate)
    actual_prevalence = (sim.population.population.simple_disease == 'sick').sum() / len(sim.population.population)

    assert np.isclose(expected_prevalence, actual_prevalence)
