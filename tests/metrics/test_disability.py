# TODO: Review for useful tests later.
# import pytest
# import numpy as np
# import pandas as pd
#
# from vivarium.testing_utilities import build_table, TestPopulation
# from vivarium.interface.interactive import setup_simulation
#
# from vivarium_public_health.population import Mortality
# from vivarium_public_health.disease import ExcessMortalityState, DiseaseModel, DiseaseState
# from vivarium_public_health.metrics import Disability
#
#
# def set_up_test_parameters(base_config, flu=False, mumps=False, deadly=False):
#     """
#     Sets up a simulation with specified disease states
#
#     flu: bool
#         If true, include an excess mortality state for flu
#         If false, do not include an excess mortality state for flu
#
#     mumps: bool
#         If true, include an excess mortality state for mumps
#         If false, do not include an excess mortality state for mumps
#     """
#     year_start = base_config.time.start.year
#     year_end = base_config.time.end.year
#     n_simulants = 1000
#
#     asymp_data_funcs = {'prevalence': lambda _, __: build_table(1.0, year_start-1, year_end,
#                                                                 ['age', 'year', 'sex', 'value']),
#                         'disability_weight': lambda _, __: 0.0,
#                         'dwell_time': lambda _, __: pd.Timedelta(days=1),
#                         'excess_mortality_rate': lambda _, __: build_table(0, year_start-1, year_end)}
#
#     asymptomatic_disease_state = ExcessMortalityState('asymptomatic', get_data_functions=asymp_data_funcs)
#     asymptomatic_disease_model = DiseaseModel('asymptomatic',
#                                               states=[asymptomatic_disease_state],
#                                               initial_state=asymptomatic_disease_state,
#                                               get_data_functions={
#                                                   'csmr': lambda _, __: build_table(0, year_start-1, year_end)})
#     disability = Disability()
#     components = [TestPopulation(), asymptomatic_disease_model, disability]
#
#     if flu:
#         flu_data_funcs = {'prevalence': lambda _, __: build_table(1.0, year_start-1, year_end,
#                                                                   ['age', 'year', 'sex', 'value']),
#                           'disability_weight': lambda _, __: 0.2,
#                           'dwell_time': lambda _, __: pd.Timedelta(days=1),
#                           'excess_mortality_rate': lambda _, __: build_table(0, year_start-1, year_end)}
#         flu = ExcessMortalityState('flu', get_data_functions=flu_data_funcs)
#         flu_model = DiseaseModel('flu', states=[flu],
#                                  initial_state=flu,
#                                  get_data_functions={'csmr': lambda _, __: build_table(0, year_start-1, year_end)})
#         components.append(flu_model)
#
#     if mumps:
#         mumps_data_funcs = {'prevalence': lambda _, __: build_table(1.0, year_start-1, year_end,
#                                                                     ['age', 'year', 'sex', 'value']),
#                             'disability_weight': lambda _, __: 0.4,
#                             'dwell_time': lambda _, __: pd.Timedelta(days=1),
#                             'excess_mortality_rate': lambda _, __: build_table(0, year_start-1, year_end)}
#         mumps = ExcessMortalityState('mumps', get_data_functions=mumps_data_funcs)
#         mumps_model = DiseaseModel('mumps', states=[mumps],
#                                    initial_state=mumps,
#                                    get_data_functions={'csmr': lambda _, __: build_table(0, year_start-1, year_end)})
#         components.append(mumps_model)
#
#     if deadly:
#         deadly_data_funcs = {'prevalence': lambda _, __: build_table(0.1, year_start-1, year_end,
#                                                                      ['age', 'year', 'sex', 'value']),
#                              'disability_weight': lambda _, __: 0.4,
#                              'dwell_time': lambda _, __: pd.Timedelta(days=1),
#                              'excess_mortality_rate': lambda _, __: build_table(0.005, year_start-1, year_end)}
#         deadly = ExcessMortalityState('deadly', get_data_functions=deadly_data_funcs)
#         healthy = DiseaseState('healthy', get_data_functions=deadly_data_funcs)
#         deadly_model = DiseaseModel('deadly', initial_state=healthy,
#                                     states=[deadly, healthy],
#                                     get_data_functions={
#                                         'csmr': lambda _, __: build_table(0.0005, year_start-1, year_end)
#                                     })
#         components.append(deadly_model)
#         components.append(Mortality())
#
#     base_config.update({'population': {'population_size': n_simulants}})
#     simulation = setup_simulation(components, base_config)
#
#     return simulation, disability
#
#
# def test_that_ylds_are_0_at_sim_beginning(base_config):
#     simulation, disability = set_up_test_parameters(base_config)
#     ylds = int(simulation.get_value('metrics')(simulation.get_population().index)['years_lived_with_disability'])
#     assert ylds == 0
#
#
# def test_that_healthy_people_dont_accrue_disability_weights(base_config):
#     simulation, disability = set_up_test_parameters(base_config)
#     simulation.run_for(duration=pd.Timedelta(days=365))
#     pop_size = len(simulation.get_population())
#     ylds = simulation.get_value('metrics')(simulation.get_population().index)['years_lived_with_disability']
#     assert np.isclose(ylds, pop_size * 0.0, rtol=0.01)
#
#
# def test_single_disability_weight(base_config):
#     simulation, disability = set_up_test_parameters(base_config, flu=True)
#     flu_dw = 0.2
#     simulation.run_for(duration=pd.Timedelta(days=365))
#     pop_size = len(simulation.get_population())
#     ylds = simulation.get_value('metrics')(simulation.get_population().index)['years_lived_with_disability']
#     assert np.isclose(ylds, pop_size * flu_dw, rtol=0.01)
#
#
# def test_joint_disability_weight(base_config):
#     simulation, disability = set_up_test_parameters(base_config, flu=True, mumps=True)
#     flu_dw = 0.2
#     mumps_dw = 0.4
#     simulation.run_for(duration=pd.Timedelta(days=365))
#     pop_size = len(simulation.get_population())
#     ylds = simulation.get_value('metrics')(simulation.get_population().index)['years_lived_with_disability']
#     # check that JOINT disability weight is correctly calculated
#     assert np.isclose(ylds, pop_size * (1-(1-flu_dw)*(1-mumps_dw)), rtol=0.01)
#
#
# @pytest.mark.skip(reason="Error in way csmr is being computed when using dataframes with inconsistent ages.")
# def test_dead_people_dont_accrue_disability(base_config):
#     simulation, disability = set_up_test_parameters(base_config, deadly=True)
#     simulation.run_for(duration=pd.Timedelta(days=365))
#     pop = simulation.get_population()
#     dead = pop[pop.alive == 'dead']
#     assert len(dead) > 0
#     assert np.all(disability.disability_weight(dead.index) == 0)
