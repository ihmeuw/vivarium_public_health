import py.test

from datetime import timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.stats import norm

from ceam_tests.util import setup_simulation, pump_simulation, build_table, generate_test_population
from ceam.interpolation import Interpolation
from ceam.framework.util import from_yearly
from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns

from ceam_inputs.gbd_mapping import risk_factors, causes

from ceam_public_health.components.risks.base_risk import RiskEffect, continuous_exposure_effect, categorical_exposure_effect, CategoricalRiskComponent, ContinuousRiskComponent, correlated_propensity

def test_RiskEffect():
    time_step = timedelta(days=30.5)
    test_exposure = [0]
    def test_function(rates, rr):
        return rates * (rr.values**test_exposure[0])
    effect = RiskEffect(build_table(1.01), build_table(0.01), causes.heart_attack, test_function)

    simulation = setup_simulation([generate_test_population, effect])

    # This one should be effected by our RiskEffect
    rates = simulation.values.get_rate('incidence_rate.'+causes.heart_attack.name)
    rates.source = simulation.tables.build_table(build_table(0.01))

    # This one should not
    other_rates = simulation.values.get_rate('incidence_rate.some_other_cause')
    other_rates.source = simulation.tables.build_table(build_table(0.01))

    assert np.all(rates(simulation.population.population.index) == from_yearly(0.01, time_step))
    assert np.all(other_rates(simulation.population.population.index) == from_yearly(0.01, time_step))

    test_exposure[0] = 1

    assert np.all(rates(simulation.population.population.index) == from_yearly(0.0101, time_step))
    assert np.all(other_rates(simulation.population.population.index) == from_yearly(0.01, time_step))

def make_dummy_column(name, initial_value):
    @listens_for('initialize_simulants')
    @uses_columns([name])
    def make_column(event):
        event.population_view.update(pd.Series(initial_value, index=event.index, name=name))
    return make_column

def test_continuous_exposure_effect():
    risk = risk_factors.systolic_blood_pressure
    exposure_function = continuous_exposure_effect(risk)

    simulation = setup_simulation([generate_test_population, make_dummy_column(risk.name+'_exposure', risk.tmrl), exposure_function])

    rates = pd.Series(0.01, index=simulation.population.population.index)
    rr = pd.Series(1.01, index=simulation.population.population.index)

    assert np.all(exposure_function(rates, rr) == 0.01)

    simulation.population.get_view([risk.name+'_exposure']).update(pd.Series(risk.tmrl + 50, index=simulation.population.population.index))

    expected_value = 0.01 * (1.01 ** (((risk.tmrl + 50) - risk.tmrl) / risk.scale))

    assert np.allclose(exposure_function(rates, rr), expected_value)

def test_categorical_exposure_effect():
    risk = risk_factors.systolic_blood_pressure
    exposure_function = categorical_exposure_effect(risk)

    simulation = setup_simulation([generate_test_population, make_dummy_column(risk.name+'_exposure', False), exposure_function])

    rates = pd.Series(0.01, index=simulation.population.population.index)
    rr = pd.DataFrame({'cat1': 1.01}, index=simulation.population.population.index)

    assert np.all(exposure_function(rates, rr) == 0.01)

    simulation.population.get_view([risk.name+'_exposure']).update(pd.Series(True, index=simulation.population.population.index))

    assert np.allclose(exposure_function(rates, rr), 0.0101)

@patch('ceam_public_health.components.risks.base_risk.inputs')
def test_CategoricalRiskComponent(inputs_mock):
    time_step = timedelta(days=30.5)
    risk = risk_factors.smoking
    inputs_mock.get_exposures.side_effect = lambda *args, **kwargs: build_table(0.5, ['age', 'year', 'sex', 'cat1', 'cat2'])
    inputs_mock.get_relative_risks.side_effect = lambda *args, **kwargs: build_table([1.01, 0], ['age', 'year', 'sex', 'cat1', 'cat2'])
    inputs_mock.get_pafs.side_effect = lambda *args, **kwargs: build_table(1)

    component = CategoricalRiskComponent(risk)

    simulation = setup_simulation([generate_test_population, component], 100000)
    pump_simulation(simulation, iterations=1)

    incidence_rate = simulation.values.get_rate('incidence_rate.'+risk.effected_causes[0].name)
    incidence_rate.source = simulation.tables.build_table(build_table(0.01))
    paf = simulation.values.get_rate('paf.'+risk.effected_causes[-1].name)

    assert np.isclose(simulation.population.population[risk.name+'_exposure'].sum() / len(simulation.population.population), 0.5, rtol=0.01)
    expected_exposed_value = 0.01 * 1.01
    expected_unexposed_value = 0.01

    exposed_index = simulation.population.population.index[simulation.population.population[risk.name+'_exposure']]
    unexposed_index = simulation.population.population.index[~simulation.population.population[risk.name+'_exposure']]

    assert np.allclose(incidence_rate(exposed_index), from_yearly(expected_exposed_value, time_step))
    assert np.allclose(incidence_rate(unexposed_index), from_yearly(expected_unexposed_value, time_step))

@patch('ceam_public_health.components.risks.base_risk.inputs')
def test_ContinuousRiskComponent(inputs_mock):
    time_step = timedelta(days=30.5)
    risk = risk_factors.systolic_blood_pressure
    inputs_mock.get_exposures.side_effect = lambda *args, **kwargs: build_table(0.5)
    inputs_mock.get_relative_risks.side_effect = lambda *args, **kwargs: build_table(1.01)
    inputs_mock.get_pafs.side_effect = lambda *args, **kwargs: build_table(1)

    def loader(builder):
        dist = Interpolation(
                build_table([130, 0.000001], ['age', 'year', 'sex', 'mean', 'std']),
                ['sex'],
                ['age', 'year'],
                func=lambda parameters: norm(loc=parameters['mean'], scale=parameters['std']).ppf)
        return builder.lookup(dist)

    component = ContinuousRiskComponent(risk, loader)

    simulation = setup_simulation([generate_test_population, component], 100000)
    pump_simulation(simulation, iterations=1)

    incidence_rate = simulation.values.get_rate('incidence_rate.'+risk.effected_causes[0].name)
    incidence_rate.source = simulation.tables.build_table(build_table(0.01))
    paf = simulation.values.get_rate('paf.'+risk.effected_causes[-1].name)

    assert np.allclose(simulation.population.population[risk.name+'_exposure'], 130, rtol=0.001)

    expected_value = 0.01 * (1.01**((130 - 112) / 10))

    assert np.allclose(incidence_rate(simulation.population.population.index), from_yearly(expected_value, time_step), rtol=0.001)

@patch('ceam_public_health.components.risks.base_risk.load_matrices')
def test_correlated_propensity(correlation_loader_mock):
    correlation_matrix = pd.DataFrame({
        'systolic_blood_pressure':    [1,0.282213017344475,0.110525231808424,0.130475437755401,0.237914389663941],
        'body_mass_index':            [0.282213017344475,1,0.0928986519575119,-0.119147761153339,0.212531763837137],
        'cholesterol':                [0.110525231808424,0.0928986519575119,1,0.175454370605231,0.0476387962101613],
        'smoking':                    [0.130475437755401,-0.119147761153339,0.175454370605231,1,0.0770317213079334],
        'fasting_plasma_glucose':     [0.237914389663941,0.212531763837137,0.0476387962101613,0.0770317213079334,1],
        'risk_factor':                ['systolic_blood_pressure', 'body_mass_index', 'cholesterol', 'smoking', 'fasting_plasma_glucose',],
        })
    correlation_matrix['age'] = 30
    correlation_matrix['sex'] = 'Male'
    correlation_loader_mock.return_value = correlation_matrix

    pop = pd.DataFrame({'age': [30]*100000, 'sex': ['Male']*100000})

    propensities = []
    for risk in [
            risk_factors.systolic_blood_pressure,
            risk_factors.body_mass_index,
            risk_factors.cholesterol,
            risk_factors.smoking,
            risk_factors.fasting_plasma_glucose]:
        propensities.append(correlated_propensity(pop, risk))

    matrix = np.corrcoef(np.array(propensities))
    assert np.allclose(correlation_matrix[['systolic_blood_pressure', 'body_mass_index', 'cholesterol', 'smoking', 'fasting_plasma_glucose',]].values, matrix, rtol=0.15)
