import pytest

import numpy as np
import pandas as pd

from ceam_tests.util import setup_simulation, pump_simulation

from ceam import config

from ceam_inputs import get_continuous, get_proportion
from ceam_public_health.components.base_population import generate_base_population, assign_location
from ceam_public_health.components.risks.base_risk import ContinuousRiskComponent, CategoricalRiskComponent

config.simulation_parameters.location_id = 180
config.simulation_parameters.year_start = 1990
config.simulation_parameters.pop_age_start = 0
config.simulation_parameters.pop_age_end = 125

RISKS = [
        [ContinuousRiskComponent(
                       "systolic_blood_pressure",
                       "ceam_public_health.components.risks.blood_pressure.distribution_loader",
                       "ceam_public_health.components.risks.blood_pressure.exposure_function"),
        [get_continuous, 2547]
         ],
        [ContinuousRiskComponent(
                       "fasting_plasma_glucose",
                       "ceam_public_health.components.risks.fasting_plasma_glucose.distribution_loader"),
        [get_continuous, 2545]
        ],
        [ContinuousRiskComponent(
                       "cholesterol",
                       "ceam_public_health.components.risks.cholesterol.distribution_loader"),
        [get_continuous, 2546]
        ],
        [ContinuousRiskComponent(
                       "body_mass_index",
                       "ceam_public_health.components.risks.body_mass_index.distribution_loader"),
        [get_continuous, 2548]
        ],
        [CategoricalRiskComponent("smoking"),
        [get_proportion, 8941]
        ],
        [CategoricalRiskComponent("household_air_polution"),
        [get_proportion, 2511]
        ]
]

PROBES = [32.5, 42.5, 52.5, 62.5, 72.5, 82.5]

@pytest.fixture(scope="module")
def simulation():
    components = [r[0] for r in RISKS]
    simulation = setup_simulation([generate_base_population, assign_location] + components, 2000000)
    pump_simulation(simulation, iterations=1)
    return simulation

@pytest.mark.slow
@pytest.mark.parametrize('component, mean_model', RISKS)
def test_risk_means(simulation, component, mean_model):
    '''
    Note
    ----
    This isn't quite the right place for this test since it depends heavily on the
    details of GBD. But right now there isn't really a better place for it.
    '''
    means = mean_model[0](mean_model[1]).query('year == 1990')
    mean_column = list(means.columns.difference(['age','sex','year']))[0]

    for sex in ['Male', 'Female']:
        for age_mid in PROBES:
            low_age = age_mid - 2.5
            high_age = age_mid + 2.5
            expected_mean = means.query('age == @age_mid and sex == @sex')[mean_column]
            observed_mean = simulation.population.population.query('age > @low_age and age <= @high_age and sex == @sex')[component._risk.name+'_exposure'].mean()
            message = 'Out of bounds for {} at age {}'.format(component._risk.name, age_mid)
            if mean_model[0] is get_proportion:
                # Categorical. Use 1 percentage point difference threshold
                assert np.isclose(observed_mean, expected_mean, atol=0.01), message
            else:
                # Continuous. Use 1 percent difference
                assert np.isclose(observed_mean, expected_mean, rtol=0.01), message
