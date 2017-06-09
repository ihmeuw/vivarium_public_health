import os
from datetime import timedelta

import numpy as np
import pytest

from ceam import config
from ceam_tests.util import setup_simulation, pump_simulation, generate_test_population

from ceam_public_health.components.risks import distributions, exposures
from ceam_public_health.components.risks import ContinuousRiskComponent
from ceam_public_health.components.base_population import age_simulants

np.random.seed(100)


def setup():
    try:
        config.reset_layer('override', preserve_keys=['input_data.intermediary_data_cache_path',
                                                      'input_data.auxiliary_data_folder'])
    except KeyError:
        pass
    config.simulation_parameters.set_with_metadata('year_start', 1990, layer='override',
                                                   source=os.path.realpath(__file__))
    config.simulation_parameters.set_with_metadata('year_end', 2010, layer='override',
                                                   source=os.path.realpath(__file__))
    config.simulation_parameters.set_with_metadata('time_step', 30.5, layer='override',
                                                   source=os.path.realpath(__file__))


@pytest.mark.slow
def test_basic_SBP_bounds():
    simulation = setup_simulation([
        generate_test_population,
        age_simulants,
        ContinuousRiskComponent('high_systolic_blood_pressure', distributions.sbp, exposures.sbp)], 1000)

    sbp_mean = 138  # Mean across all demographics
    sbp_std = 15  # Standard deviation across all demographics
    interval = sbp_std * 4
    pump_simulation(simulation, iterations=1)  # Get blood pressure stabilized

    # We don't model SBP for simulants 27 and under, so exclude those from some of our tests
    idx = simulation.population.population.age > 27

    # Check that no one is wildly out of range
    assert ((simulation.population.population[idx].high_systolic_blood_pressure_exposure > sbp_mean + 2*interval)
            | (simulation.population.population[idx].high_systolic_blood_pressure_exposure < sbp_mean-interval)).sum() == 0

    initial_mean_sbp = simulation.population.population[idx].high_systolic_blood_pressure_exposure.mean()

    pump_simulation(simulation, duration=timedelta(days=5*365))

    # Check that blood pressure goes up over time as our cohort ages
    assert simulation.population.population[idx].high_systolic_blood_pressure_exposure.mean() > initial_mean_sbp
    # And that there's still no one wildly out of bounds
    assert ((simulation.population.population[idx].high_systolic_blood_pressure_exposure > sbp_mean + 2*interval)
            | (simulation.population.population[idx].high_systolic_blood_pressure_exposure < sbp_mean - interval)).sum() == 0


# TODO: The change to risk deleted incidence rates breaks these tests. We need a new way of checking face validity
#@pytest.mark.parametrize('condition_module, rate_label', [(heart_disease_factory(),
# 'heart_attack'), (stroke_factory(), 'hemorrhagic_stroke'), (stroke_factory(), 'ischemic_stroke'), ])
#@pytest.mark.slow
#def test_blood_pressure_effect_on_incidince(condition_module, rate_label):
#    bp_module = BloodPressureModule()
#    simulation = simulation_factory([bp_module, condition_module])
#
#    pump_simulation(simulation, iterations=1) # Get blood pressure stablaized
#    simulation.remove_children([bp_module])
#
#    # Base incidence rate without blood pressure
#    base_incidence = simulation.incidence_rates(simulation.population, rate_label)
#
#    simulation.add_children([bp_module])
#
#    # Get incidence including the effect of blood pressure
#    bp_incidence = simulation.incidence_rates(simulation.population, rate_label)
#
#    # Blood pressure should only increase rates
#    assert base_incidence.mean() < bp_incidence.mean()
#
#    pump_simulation(simulation, duration=timedelta(days=5*365))
#
#    # Increase in incidence should rise over time as the cohort ages and SBP increases
#    assert bp_incidence.mean() < simulation.incidence_rates(simulation.population, rate_label).mean()
