# import pytest
#
# import numpy as np
# import pandas as pd
# from scipy.stats import norm
#
# from vivarium.testing_utilities import TestPopulation, metadata
# from vivarium.interface.interactive import initialize_simulation
# from vivarium_public_health.risks.base_risk import Risk
#
#
# @pytest.mark.parametrize('propensity', [0.00001, 0.5, 0.99])
# def test_propensity_effect(propensity, mocker, continuous_risk, base_config, base_plugins):
#     population_size = 1000
#
#     rf, risk_data = continuous_risk
#     base_config.update({'population': {'population_size': population_size}}, **metadata(__file__))
#     sim = initialize_simulation([TestPopulation(), rf], input_config=base_config, plugin_config=base_plugins)
#     for key, value in risk_data.items():
#         sim.data.write(f'risk_factor.test_risk.{key}', value)
#
#     sim.setup()
#     propensity_pipeline = mocker.Mock()
#     sim.values.register_value_producer('test_risk.propensity', source=propensity_pipeline)
#     propensity_pipeline.side_effect = lambda index: pd.Series(propensity, index=index)
#
#     expected_value = norm(loc=130, scale=15).ppf(propensity)
#
#     assert np.allclose(rf.exposure(sim.get_population().index), expected_value)
#
#
# def test_Risk_config_data(base_config, base_plugins):
#     exposure_level = 0.8  # default is one
#     dummy_risk = Risk("risk_factor.test_risk")
#     base_config.update({'test_risk': {'exposure': exposure_level}}, layer='override')
#
#     simulation = initialize_simulation([TestPopulation(), dummy_risk],
#                                        input_config=base_config, plugin_config=base_plugins)
#     simulation.setup()
#
#     # Make sure dummy exposure is being used
#     exp = simulation.values.get_value('test_risk.exposure')(simulation.get_population().index)
#     exposed_proportion = (exp == 'cat1').sum() / len(exp)
#     assert np.isclose(exposed_proportion, exposure_level, atol=0.005)  # population is 1000
#
#     # Make sure value was correctly pulled from config
#     sim_exposure_level = simulation.values.get_value('test_risk.exposure_parameters')(simulation.get_population().index)
#     assert np.all(sim_exposure_level == exposure_level)
