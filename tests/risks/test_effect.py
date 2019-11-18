# import numpy as np
# import pandas as pd
#
# from vivarium.framework.utilities import from_yearly
# from vivarium.testing_utilities import build_table, TestPopulation
# from vivarium.interface.interactive import initialize_simulation
#
# from vivarium_public_health.disease import RateTransition
# from vivarium_public_health.risks.effect import RiskEffect
# from vivarium_public_health.risks.base_risk import Risk
#
#
# def test_incidence_rate_risk_effect(base_config, base_plugins, mocker):
#     year_start = base_config.time.start.year
#     year_end = base_config.time.end.year
#     time_step = pd.Timedelta(days=base_config.time.step_size)
#     test_exposure = [0]
#
#     def test_function(rates_, rr):
#         return rates_ * (rr.values**test_exposure[0])
#
#     r = 'test_risk'
#     d = 'test_cause'
#     rf = Risk(f'risk_factor.{r}')
#     effect_data_functions = {
#         'rr': lambda *args: build_table([1.01, 'per_unit', d, 'incidence_rate'], year_start, year_end,
#                                         ('age', 'year', 'sex', 'value', 'parameter', 'cause', 'affected_measure')),
#         'paf': lambda *args: build_table([0.01, d, 'incidence_rate'], year_start, year_end,
#                                          ('age', 'year', 'sex', 'value', 'cause', 'affected_measure')),
#     }
#
#     effect = RiskEffect(f'risk_factor.{r}', f'cause.{d}.incidence_rate', effect_data_functions)
#
#     simulation = initialize_simulation([TestPopulation(), effect], input_config=base_config, plugin_config=base_plugins)
#
#     simulation.data.write("risk_factor.test_risk.distribution", "dichotomous")
#     simulation.values.register_value_producer("test_risk.exposure", mocker.Mock())
#
#     simulation.setup()
#
#     effect.exposure_effect = test_function
#
#     # This one should be affected by our RiskEffect
#     rates = simulation.values.register_rate_producer('test_cause.incidence_rate')
#     rates.source = simulation.tables.build_table(build_table(0.01, year_start, year_end),
#                                                  key_columns=('sex',),
#                                                  parameter_columns=[('age', 'age_start', 'age_end'),
#                                                                     ('year', 'year_start', 'year_end')],
#                                                  value_columns=None)
#
#     # This one should not
#     other_rates = simulation.values.register_rate_producer('some_other_cause.incidence_rate')
#     other_rates.source = simulation.tables.build_table(build_table(0.01, year_start, year_end),
#                                                        key_columns=('sex',),
#                                                        parameter_columns=[('age', 'age_start', 'age_end'),
#                                                                           ('year', 'year_start', 'year_end')],
#                                                        value_columns=None)
#
#     assert np.allclose(rates(simulation.get_population().index), from_yearly(0.01, time_step))
#     assert np.allclose(other_rates(simulation.get_population().index), from_yearly(0.01, time_step))
#
#     test_exposure[0] = 1
#
#     assert np.allclose(rates(simulation.get_population().index), from_yearly(0.0101, time_step))
#     assert np.allclose(other_rates(simulation.get_population().index), from_yearly(0.01, time_step))
#
#
# def test_risk_deletion(base_config, base_plugins, mocker):
#     year_start = base_config.time.start.year
#     year_end = base_config.time.end.year
#     time_step = pd.Timedelta(days=base_config.time.step_size)
#
#     base_rate = 0.01
#     risk_paf = 0.5
#     risk_rr = 1
#
#     rate_data_functions = {
#         'incidence_rate': lambda *args: build_table(0.01, year_start, year_end, ('age', 'year', 'sex', 'value'))
#     }
#
#     effect_data_functions = {
#         'rr': lambda *args: build_table([risk_rr, 'per_unit', 'infected','incidence_rate'], year_start, year_end,
#                                         ('age', 'year', 'sex', 'value', 'parameter', 'cause', 'affected_measure')),
#         'paf': lambda *args: build_table([risk_paf, 'infected', 'incidence_rate'], year_start, year_end,
#                                          ('age', 'year', 'sex', 'value', 'cause', 'affected_measure')),
#     }
#
#     def effect_function(rates, _):
#         return rates
#
#     transition = RateTransition(mocker.MagicMock(state_id='susceptible'),
#                                 mocker.MagicMock(state_id='infected'), rate_data_functions)
#
#     base_simulation = initialize_simulation([TestPopulation(), transition],
#                                             input_config=base_config, plugin_config=base_plugins)
#     base_simulation.setup()
#
#     incidence = base_simulation.get_value('infected.incidence_rate')
#     joint_paf = base_simulation.get_value('infected.incidence_rate.paf')
#
#     # Validate the base case
#     assert np.allclose(incidence(base_simulation.get_population().index), from_yearly(base_rate, time_step))
#     assert np.allclose(joint_paf(base_simulation.get_population().index), 0)
#
#     transition = RateTransition(mocker.MagicMock(state_id='susceptible'),
#                                 mocker.MagicMock(state_id='infected'), rate_data_functions)
#     effect = RiskEffect(f'risk_factor.bad_risk', f'cause.infected.incidence_rate', effect_data_functions)
#
#     rf_simulation = initialize_simulation([TestPopulation(), transition, effect],
#                                           input_config=base_config, plugin_config=base_plugins)
#
#     rf_simulation.data.write("risk_factor.bad_risk.distribution", "dichotomuous")
#     rf_simulation.values.register_value_producer("bad_risk.exposure", mocker.Mock())
#
#     rf_simulation.setup()
#     effect.exposure_effect = effect_function
#
#     incidence = rf_simulation.get_value('infected.incidence_rate')
#     joint_paf = rf_simulation.get_value('infected.incidence_rate.paf')
#
#     assert np.allclose(incidence(rf_simulation.get_population().index),
#                        from_yearly(base_rate * (1 - risk_paf), time_step))
#     assert np.allclose(joint_paf(rf_simulation.get_population().index), risk_paf)
#
#
# def test_continuous_exposure_effect(mocker, base_config, base_plugins, continuous_risk):
#     risk, risk_data = continuous_risk
#
#     class exposure_function_wrapper:
#
#         def setup(self, builder):
#             self.exposure_function = RiskEffect.get_exposure_effect(builder, 'test_risk', 'risk_factor',
#                                                                     risk_data['distribution'])
#
#         def __call__(self, *args, **kwargs):
#             return self.exposure_function(*args, **kwargs)
#
#     exposure_function = exposure_function_wrapper()
#
#     components = [TestPopulation(), exposure_function]
#     simulation = initialize_simulation(components, input_config=base_config, plugin_config=base_plugins)
#     for key, value in risk_data.items():
#         simulation.data.write(f'risk_factor.test_risk.{key}', value)
#
#     risk_exposure_pipeline = mocker.Mock()
#     simulation.values.register_value_producer('test_risk.exposure', source=risk_exposure_pipeline)
#     risk_exposure_pipeline.side_effect = lambda index: pd.Series(risk_data['tmrel'], index=index)
#
#     simulation.setup()
#
#     rates = pd.Series(0.01, index=simulation.get_population().index)
#     rr = pd.Series(1.01, index=simulation.get_population().index)
#
#     assert np.all(exposure_function(rates, rr) == 0.01)
#
#     simulation.values.register_value_producer('test_risk.exposure', source=risk_exposure_pipeline)
#     risk_exposure_pipeline.side_effect = lambda index: pd.Series(risk_data['tmrel']+50, index=index)
#
#     expected_value = 0.01 * (1.01 ** (((risk_data['tmrel'] + 50) - risk_data['tmrel'])
#                                       / risk_data['exposure_parameters']["scale"]))
#
#     assert np.allclose(exposure_function(rates, rr), expected_value)
#
#
# def test_categorical_exposure_effect(base_config, base_plugins, mocker):
#     risk_effect = mocker.Mock()
#     risk_effect.risk = 'test_risk'
#
#     class exposure_function_wrapper:
#         def setup(self, builder):
#             self.exposure_function = RiskEffect.get_exposure_effect(builder, 'test_risk', 'risk_factor', 'dichotomous')
#
#         def __call__(self, *args, **kwargs):
#             return self.exposure_function(*args, **kwargs)
#
#     exposure_function = exposure_function_wrapper()
#     components = [TestPopulation(), exposure_function]
#
#     simulation = initialize_simulation(components, input_config=base_config, plugin_config=base_plugins)
#
#     test_risk_exposure = mocker.Mock()
#     simulation.values.register_value_producer('test_risk.exposure', test_risk_exposure)
#     test_risk_exposure.side_effect = lambda index: pd.Series(['cat2'] * len(index), index=index)
#     simulation.data.write("risk_factor.test_risk.distribution", "dichotomous")
#     simulation.setup()
#
#     rates = pd.Series(0.01, index=simulation.get_population().index)
#     rr = pd.DataFrame({'cat1': 1.01, 'cat2': 1}, index=simulation.get_population().index)
#
#     assert np.all(exposure_function(rates, rr) == 0.01)
#
#     test_risk_exposure.side_effect = lambda index: pd.Series(['cat1'] * len(index), index=index)
#     simulation.step()
#
#     rates = pd.Series(0.01, index=simulation.get_population().index)
#     rr = pd.DataFrame({'cat1': 1.01, 'cat2': 1}, index=simulation.get_population().index)
#
#     assert np.allclose(exposure_function(rates, rr), 0.0101)
#
#
# def test_CategoricalRiskComponent_dichotomous_case(base_config, base_plugins, dichotomous_risk):
#     time_step = pd.Timedelta(days=base_config.time.step_size)
#     risk, risk_data = dichotomous_risk
#     affected_causes = risk_data['affected_causes']
#     risk_effects = [RiskEffect(f'risk_factor.{risk._risk}', f'cause.{ac}.incidence_rate') for ac in affected_causes]
#
#     base_config.update({'population': {'population_size': 100000}}, layer='override')
#
#     simulation = initialize_simulation([TestPopulation(), risk] + risk_effects,
#                                        input_config=base_config, plugin_config=base_plugins)
#     for key, value in risk_data.items():
#         simulation.data.write(f'risk_factor.test_risk.{key}', value)
#
#     simulation.setup()
#
#     incidence_rate = simulation.values.register_rate_producer(affected_causes[0]+'.incidence_rate')
#     incidence_rate.source = simulation.tables.build_table(risk_data['incidence_rate'], key_columns=('sex',),
#                                                           parameter_columns=[('age', 'age_start', 'age_end'),
#                                                                              ('year', 'year_start', 'year_end')],
#                                                           value_columns=None)
#
#     categories = simulation.values.get_value('test_risk.exposure')(simulation.get_population().index)
#     assert np.isclose(categories.value_counts()['cat1'] / len(simulation.get_population()), 0.5, rtol=0.01)
#
#     expected_exposed_value = 0.01 * 1.01
#     expected_unexposed_value = 0.01
#
#     exposed_index = categories[categories == 'cat1'].index
#     unexposed_index = categories[categories == 'cat2'].index
#
#     assert np.allclose(incidence_rate(exposed_index), from_yearly(expected_exposed_value, time_step))
#     assert np.allclose(incidence_rate(unexposed_index), from_yearly(expected_unexposed_value, time_step))
#
#
# def test_CategoricalRiskComponent_polytomous_case(base_config, base_plugins, polytomous_risk):
#     time_step = pd.Timedelta(days=base_config.time.step_size)
#     risk, risk_data = polytomous_risk
#     affected_causes = risk_data['affected_causes']
#
#     risk_effects = [RiskEffect(f'risk_factor.{risk._risk}', f'cause.{ac}.incidence_rate') for ac in affected_causes]
#
#     base_config.update({'population': {'population_size': 100000}}, layer='override')
#     simulation = initialize_simulation([TestPopulation(), risk] + risk_effects,
#                                        input_config=base_config, plugin_config=base_plugins)
#
#     for key, value in risk_data.items():
#         simulation.data.write(f'risk_factor.test_risk.{key}', value)
#
#     simulation.setup()
#
#     incidence_rate = simulation.values.register_rate_producer(affected_causes[0]+'.incidence_rate')
#     incidence_rate.source = simulation.tables.build_table(risk_data['incidence_rate'],
#                                                           key_columns=('sex',),
#                                                           parameter_columns=[('age', 'age_start', 'age_end'),
#                                                                              ('year', 'year_start', 'year_end')],
#                                                           value_columns=None)
#
#     categories = simulation.values.get_value('test_risk.exposure')(simulation.get_population().index)
#
#     for category in ['cat1', 'cat2', 'cat3', 'cat4']:
#         assert np.isclose(categories.value_counts()[category] / len(simulation.get_population()), 0.25, rtol=0.02)
#
#     expected_exposed_value = 0.01 * np.array([1.02, 1.03, 1.01])
#
#     for cat, expected in zip(['cat1', 'cat2', 'cat3', 'cat4'], expected_exposed_value):
#         exposed_index = categories[categories == cat].index
#         assert np.allclose(incidence_rate(exposed_index), from_yearly(expected, time_step), rtol=0.01)
#
#
# def test_ContinuousRiskComponent(continuous_risk, base_config, base_plugins):
#     year_start, year_end = base_config.time.start.year, base_config.time.end.year
#     time_step = pd.Timedelta(days=base_config.time.step_size)
#     risk, risk_data = continuous_risk
#     risk_data['exposure_standard_deviation'] = build_table(0.0001, year_start, year_end, ('age', 'year', 'sex', 'value'))
#     risk_effects = [RiskEffect(f'risk_factor.{risk._risk}', f'cause.{ac}.incidence_rate') for ac in risk_data['affected_causes']]
#
#     base_config.update({'population': {'population_size': 100000}}, layer='override')
#     simulation = initialize_simulation([TestPopulation(), risk] + risk_effects,
#                                        input_config=base_config, plugin_config=base_plugins)
#     for key, value in risk_data.items():
#         simulation.data.write(f'risk_factor.test_risk.{key}', value)
#
#     simulation.setup()
#     affected_causes = risk_data['affected_causes']
#
#     incidence_rate = simulation.values.register_rate_producer(affected_causes[0]+'.incidence_rate',
#                                                               source=lambda index: pd.Series(0.01, index=index))
#
#     exposure = simulation.values.get_value('test_risk.exposure')
#
#     assert np.allclose(exposure(simulation.get_population().index), 130, rtol=0.001)
#
#     expected_value = 0.01 * (1.01**((130 - 112) / 10))
#
#     assert np.allclose(incidence_rate(simulation.get_population().index),
#                        from_yearly(expected_value, time_step), rtol=0.001)
#
#
# def test_exposure_params_risk_effect_dichotomous(base_config, base_plugins, dichotomous_risk, coverage_gap):
#     affected_risk, risk_data = dichotomous_risk
#     coverage_gap, cg_data = coverage_gap
#     rf_exposed = 0.5
#     rr = 2 # rr between cg/affected_risk
#
#     base_config.update({'population': {'population_size': 100000}}, layer='override')
#     affected_risk = Risk('risk_factor.test_risk')
#
#     # start with the only risk factor without indirect effect from coverage_gap
#     simulation = initialize_simulation([TestPopulation(), affected_risk],
#                                        input_config=base_config, plugin_config=base_plugins)
#
#     for key, value in risk_data.items():
#         simulation.data.write(f'risk_factor.test_risk.{key}', value)
#
#     simulation.setup()
#
#     pop = simulation.get_population()
#     exposure = simulation.values.get_value('test_risk.exposure')
#     assert np.isclose(rf_exposed, exposure(pop.index).value_counts()['cat1']/len(pop), rtol=0.01)
#
#     # add the coverage gap which should change the exposure of test risk
#     risk_effects = [RiskEffect(f'coverage_gap.{coverage_gap._risk}',
#                                f'risk_factor.{rf}.exposure_parameters') for rf in cg_data['affected_risk_factors']]
#
#     simulation = initialize_simulation([TestPopulation(), affected_risk, coverage_gap] + risk_effects,
#                                        input_config=base_config, plugin_config=base_plugins)
#
#     for key, value in risk_data.items():
#         simulation.data.write(f'risk_factor.test_risk.{key}', value)
#
#     for key, value in cg_data.items():
#         simulation.data.write(f'coverage_gap.test_coverage_gap.{key}', value)
#
#     simulation.setup()
#
#     pop = simulation.get_population()
#     rf_exposure = simulation.values.get_value('test_risk.exposure')(pop.index)
#
#     # proportion of simulants exposed to each category of affected risk stays same
#     assert np.isclose(rf_exposed, rf_exposure.value_counts()['cat1']/len(pop), rtol=0.01)
#
#     # compute relative risk to test whether it matches with the given relative risk
#     cg_exposure = simulation.values.get_value('test_coverage_gap.exposure')(pop.index)
#
#     cg_exposed = cg_exposure == 'cat1'
#     rf_exposed = rf_exposure == 'cat1'
#
#     affected_by_cg = rf_exposed & cg_exposed
#     not_affected_by_cg = rf_exposed & ~cg_exposed
#
#     computed_rr = (len(pop[affected_by_cg])/len(pop[cg_exposed])) / (len(pop[not_affected_by_cg])/len(pop[~cg_exposed]))
#     assert np.isclose(computed_rr, rr, rtol=0.01)
#
#
# def test_RiskEffect_config_data(base_config, base_plugins):
#     dummy_risk = Risk("risk_factor.test_risk")
#     dummy_effect = RiskEffect("risk_factor.test_risk", "cause.test_cause.incidence_rate")
#     year_start = base_config.time.start.year
#     year_end = base_config.time.end.year
#     time_step = pd.Timedelta(days=base_config.time.step_size)
#
#     base_config.update({'test_risk': {'exposure': 1}}, layer='override')
#     base_config.update({'effect_of_test_risk_on_test_cause': {'incidence_rate': 50}})
#     simulation = initialize_simulation([TestPopulation(), dummy_risk, dummy_effect],
#                                        input_config=base_config, plugin_config=base_plugins)
#
#     simulation.setup()
#
#     # make sure our dummy exposure value is being properly used
#     exp = simulation.values.get_value('test_risk.exposure')(simulation.get_population().index)
#     assert((exp == 'cat1').all())
#
#     # This one should be affected by our DummyRiskEffect
#     rates = simulation.values.register_rate_producer('test_cause.incidence_rate',
#                                                      source=lambda index: pd.Series(0.01, index=index))
#
#     # This one should not
#     other_rates = simulation.values.register_rate_producer('some_other_cause.incidence_rate',
#                                                            source=lambda index: pd.Series(0.01, index=index))
#
#     assert np.allclose(rates(simulation.get_population().index), from_yearly(0.01, time_step)*50)
#     assert np.allclose(other_rates(simulation.get_population().index), from_yearly(0.01, time_step))
#
#
# def test_RiskEffect_excess_mortality(base_config, base_plugins):
#     dummy_risk = Risk("risk_factor.test_risk")
#     dummy_effect = RiskEffect("risk_factor.test_risk", "cause.test_cause.excess_mortality_rate")
#     time_step = pd.Timedelta(days=base_config.time.step_size)
#
#     base_config.update({'test_risk': {'exposure': 1}}, layer='override')
#     base_config.update({'effect_of_test_risk_on_test_cause': {'excess_mortality_rate': 50}})
#
#     simulation = initialize_simulation([TestPopulation(), dummy_risk, dummy_effect],
#                                        input_config=base_config, plugin_config=base_plugins)
#     simulation.setup()
#
#     em = simulation.values.register_rate_producer('test_cause.excess_mortality_rate',
#                                                   source=lambda index: pd.Series(0.1, index=index))
#
#     assert np.allclose(from_yearly(0.1, time_step)*50, em(simulation.get_population().index))
