import pytest

import numpy as np
import pandas as pd
from scipy.stats import norm

from vivarium.testing_utilities import TestPopulation, metadata
from vivarium.interface.interactive import initialize_simulation


@pytest.mark.parametrize('propensity', [0.00001, 0.5, 0.99])
def test_propensity_effect(propensity, mocker, continuous_risk, base_config, base_plugins):
    population_size = 1000

    rf, risk_data = continuous_risk
    base_config.update({'population': {'population_size': population_size}}, **metadata(__file__))
    sim = initialize_simulation([TestPopulation(), rf], input_config=base_config, plugin_config=base_plugins)
    for key, value in risk_data.items():
        sim.data.write(f'risk_factor.test_risk.{key}', value)

    sim.setup()
    propensity_pipeline = mocker.Mock()
    sim.values.register_value_producer('test_risk.propensity', source=propensity_pipeline)
    propensity_pipeline.side_effect = lambda index: pd.Series(propensity, index=index)

    expected_value = norm(loc=130, scale=15).ppf(propensity)

    assert np.allclose(rf.exposure(sim.population.population.index), expected_value)



