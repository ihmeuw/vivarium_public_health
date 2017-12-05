import os
from collections import namedtuple

import pytest
import numpy as np
import pandas as pd
from scipy.stats import norm

from vivarium.config_tree import ConfigTree
from vivarium.framework.event import listens_for
from vivarium.framework.population import uses_columns
from vivarium.framework.values import list_combiner, joint_value_post_processor
from vivarium.framework.util import from_yearly
from vivarium.interpolation import Interpolation
from vivarium.test_util import setup_simulation, pump_simulation, build_table, TestPopulation

from ceam_inputs import risk_factors, sequelae

from ceam_public_health.disease import RateTransition, DiseaseState, BaseDiseaseState
from ceam_public_health.risks.effect import continuous_exposure_effect, categorical_exposure_effect, RiskEffect
from ceam_public_health.risks.base_risk import (CategoricalRiskComponent, ContinuousRiskComponent,
                                                correlated_propensity_factory, uncorrelated_propensity)


Disease = namedtuple('Disease', 'name')
Risk = namedtuple('Risk', ['name', 'distribution'])


@pytest.fixture(scope='function')
def config(base_config):
    try:
        base_config.reset_layer('override', preserve_keys=['input_data.intermediary_data_cache_path',
                                                           'input_data.auxiliary_data_folder'])
    except KeyError:
        pass

    metadata = {'layer': 'override', 'source': os.path.realpath(__file__)}
    base_config.simulation_parameters.set_with_metadata('year_start', 1990, **metadata)
    base_config.simulation_parameters.set_with_metadata('year_end', 2010, **metadata)
    base_config.simulation_parameters.set_with_metadata('time_step', 30.5, **metadata)
    return base_config


@pytest.fixture(scope='function')
def get_exposure_mock(mocker):
    return mocker.patch('ceam_public_health.risks.base_risk.get_exposure')


@pytest.fixture(scope='function')
def load_risk_corr_mock(mocker):
    return mocker.patch('ceam_public_health.risks.base_risk.get_risk_correlation_matrices')


@pytest.fixture(scope='function')
def get_rr_mock(mocker):
    return mocker.patch('ceam_public_health.risks.effect.get_relative_risks')


@pytest.fixture(scope='function')
def get_paf_mock(mocker):
    return mocker.patch('ceam_public_health.risks.effect.get_population_attributable_fraction')


@pytest.fixture(scope='function')
def get_mf_mock(mocker):
    return mocker.patch('ceam_public_health.risks.effect.get_mediation_factors')


@pytest.fixture(scope='function')
def get_distribution_mock(mocker):
    return mocker.patch('ceam_public_health.risks.base_risk.get_distribution')


def test_RiskEffect(config):
    year_start = config.simulation_parameters.year_start
    year_end = config.simulation_parameters.year_end
    config.simulation_parameters.time_step = 30.5
    time_step = pd.Timedelta(days=30.5)
    test_exposure = [0]

    def test_function(rates_, rr):
        return rates_ * (rr.values**test_exposure[0])

    r = Risk(name='test_risk', distribution='categorical')
    d = Disease(name='test_cause')
    effect_data_functions = {
        'rr': lambda *args: build_table(1.01, year_start, year_end),
        'paf': lambda *args: build_table(0.01, year_start, year_end),
        'mf': lambda *args: 0,
    }

    effect = RiskEffect(r, d, effect_data_functions)
    effect.exposure_effect = test_function

    simulation = setup_simulation([TestPopulation(), effect], input_config=config)

    # This one should be affected by our RiskEffect
    rates = simulation.values.get_rate('test_cause.incidence_rate')
    rates.source = simulation.tables.build_table(build_table(0.01, year_start, year_end))

    # This one should not
    other_rates = simulation.values.get_rate('some_other_cause.incidence_rate')
    other_rates.source = simulation.tables.build_table(build_table(0.01, year_start, year_end))

    assert np.allclose(rates(simulation.population.population.index), from_yearly(0.01, time_step))
    assert np.allclose(other_rates(simulation.population.population.index), from_yearly(0.01, time_step))

    test_exposure[0] = 1

    assert np.allclose(rates(simulation.population.population.index), from_yearly(0.0101, time_step))
    assert np.allclose(other_rates(simulation.population.population.index), from_yearly(0.01, time_step))


def make_dummy_column(name, initial_value):
    @listens_for('initialize_simulants')
    @uses_columns([name])
    def make_column(event):
        event.population_view.update(pd.Series(initial_value, index=event.index, name=name))
    return make_column


def test_continuous_exposure_effect(config):
    risk = risk_factors.high_systolic_blood_pressure
    exposure_function = continuous_exposure_effect(risk)
    tmrel = 0.5 * (risk.tmred.max + risk.tmred.min)
    components = [TestPopulation(), make_dummy_column(risk.name+'_exposure', tmrel), exposure_function]
    simulation = setup_simulation(components, input_config=config)

    rates = pd.Series(0.01, index=simulation.population.population.index)
    rr = pd.Series(1.01, index=simulation.population.population.index)

    assert np.all(exposure_function(rates, rr) == 0.01)

    simulation.population.get_view([risk.name+'_exposure']).update(
        pd.Series(tmrel + 50, index=simulation.population.population.index))

    expected_value = 0.01 * (1.01 ** (((tmrel + 50) - tmrel) / risk.exposure_parameters.scale))

    assert np.allclose(exposure_function(rates, rr), expected_value)


def test_categorical_exposure_effect(config):
    risk = risk_factors.high_systolic_blood_pressure
    exposure_function = categorical_exposure_effect(risk)
    components = [TestPopulation(), make_dummy_column(risk.name+'_exposure', 'cat2'), exposure_function]
    simulation = setup_simulation(components, input_config=config)

    rates = pd.Series(0.01, index=simulation.population.population.index)
    rr = pd.DataFrame({'cat1': 1.01, 'cat2': 1}, index=simulation.population.population.index)

    assert np.all(exposure_function(rates, rr) == 0.01)

    simulation.population.get_view([risk.name+'_exposure']).update(
        pd.Series('cat1', index=simulation.population.population.index))

    assert np.allclose(exposure_function(rates, rr), 0.0101)


def test_CategoricalRiskComponent_dichotomous_case(get_exposure_mock, get_paf_mock,
                                                   get_rr_mock, get_mf_mock, config):
    time_step = pd.Timedelta(days=30.5)
    config.simulation_parameters.time_step = 30.5
    risk = risk_factors.smoking_prevalence_approach
    year_start = config.simulation_parameters.year_start
    year_end = config.simulation_parameters.year_end

    get_exposure_mock.side_effect = lambda *args, **kwargs: build_table(
        0.5, year_start, year_end, ['age', 'year', 'sex', 'cat1', 'cat2'])
    get_rr_mock.side_effect = lambda *args, **kwargs: build_table(
        [1.01, 1], year_start, year_end, ['age', 'year', 'sex', 'cat1', 'cat2'])

    get_paf_mock.side_effect = lambda *args, **kwargs: build_table(1, year_start, year_end)
    get_mf_mock.side_effect = lambda *args, **kwargs: 0

    component = CategoricalRiskComponent(risk)

    simulation = setup_simulation([TestPopulation(), component], 100000, input_config=config)
    pump_simulation(simulation, iterations=1)

    incidence_rate = simulation.values.get_rate(risk.affected_causes[0].name+'.incidence_rate')
    incidence_rate.source = simulation.tables.build_table(build_table(0.01, year_start, year_end))

    assert np.isclose((simulation.population.population[risk.name+'_exposure'] == 'cat1').sum()
                      / len(simulation.population.population), 0.5, rtol=0.01)

    expected_exposed_value = 0.01 * 1.01
    expected_unexposed_value = 0.01

    exposed_index = simulation.population.population.index[
        simulation.population.population[risk.name+'_exposure'] == 'cat1']
    unexposed_index = simulation.population.population.index[
        simulation.population.population[risk.name+'_exposure'] == 'cat2']

    assert np.allclose(incidence_rate(exposed_index), from_yearly(expected_exposed_value, time_step))
    assert np.allclose(incidence_rate(unexposed_index), from_yearly(expected_unexposed_value, time_step))


def test_CategoricalRiskComponent_polytomous_case(get_exposure_mock, get_rr_mock, get_paf_mock,
                                                  get_mf_mock, config):
    time_step = pd.Timedelta(days=30.5)
    config.simulation_parameters.time_step = 30.5
    year_start = config.simulation_parameters.year_start
    year_end = config.simulation_parameters.year_end

    risk = risk_factors.smoking_prevalence_approach
    get_exposure_mock.side_effect = lambda *args, **kwargs: build_table(
        0.25, year_start, year_end, ['age', 'year', 'sex', 'cat1', 'cat2', 'cat3', 'cat4'])
    get_rr_mock.side_effect = lambda *args, **kwargs: build_table(
        [1.03, 1.02, 1.01, 1], year_start, year_end, ['age', 'year', 'sex', 'cat1', 'cat2', 'cat3', 'cat4'])
    get_paf_mock.side_effect = lambda *args, **kwargs: build_table(1, year_start, year_end)
    get_mf_mock.side_effect = lambda *args, **kwargs: 0

    component = CategoricalRiskComponent(risk)

    simulation = setup_simulation([TestPopulation(), component], 100000, input_config=config)
    pump_simulation(simulation, iterations=1)

    incidence_rate = simulation.values.get_rate(risk.affected_causes[0].name+'.incidence_rate')
    incidence_rate.source = simulation.tables.build_table(build_table(0.01, year_start, year_end))

    for category in ['cat1', 'cat2', 'cat3', 'cat4']:
        assert np.isclose((simulation.population.population[risk.name+'_exposure'] == category).sum()
                          / len(simulation.population.population), 0.25, rtol=0.02)

    expected_exposed_value = 0.01 * np.array([1.02, 1.03, 1.01])

    for cat, expected in zip(['cat1', 'cat2', 'cat3', 'cat4'], expected_exposed_value):
        exposed_index = simulation.population.population.index[
            simulation.population.population[risk.name+'_exposure'] == cat]
        assert np.allclose(incidence_rate(exposed_index), from_yearly(expected, time_step), rtol=0.01)


def test_ContinuousRiskComponent(get_exposure_mock, get_rr_mock, get_paf_mock, get_mf_mock,
                                 get_distribution_mock, config):
    time_step = pd.Timedelta(days=30.5)
    year_start = config.simulation_parameters.year_start
    year_end = config.simulation_parameters.year_end
    risk = risk_factors.high_systolic_blood_pressure
    get_exposure_mock.side_effect = lambda *args, **kwargs: build_table(0.5, year_start, year_end)
    get_rr_mock.side_effect = lambda *args, **kwargs: build_table(1.01, year_start, year_end)
    get_paf_mock.side_effect = lambda *args, **kwargs: build_table(1, year_start, year_end)
    get_mf_mock.side_effect = lambda *args, **kwargs: 0

    class Distribution:
        def __init__(self, *_, **__):
            pass

        def setup(self, builder):
            data = build_table([130, 0.000001], year_start, year_end, ['age', 'year', 'sex', 'mean', 'std'])
            self.parameters = builder.lookup(data)

        def ppf(self, propensity):
            params = self.parameters(propensity.index)
            return norm(loc=params['mean'], scale=params['std']).ppf(propensity)

    get_distribution_mock.side_effect = lambda *args, **kwargs: Distribution(args, kwargs)

    component = ContinuousRiskComponent(risk)

    simulation = setup_simulation([TestPopulation(), component], 100000, input_config=config)
    pump_simulation(simulation, iterations=1)

    incidence_rate = simulation.values.get_rate(risk.affected_causes[0].name+'.incidence_rate')
    incidence_rate.source = simulation.tables.build_table(build_table(0.01, year_start, year_end))

    assert np.allclose(simulation.population.population[risk.name+'_exposure'], 130, rtol=0.001)

    expected_value = 0.01 * (1.01**((130 - 112) / 10))

    assert np.allclose(incidence_rate(simulation.population.population.index),
                       from_yearly(expected_value, time_step), rtol=0.001)


def test_propensity_effect(get_exposure_mock, get_rr_mock, get_paf_mock, get_mf_mock,
                           get_distribution_mock, config):
    year_start = config.simulation_parameters.year_start
    year_end = config.simulation_parameters.year_end
    risk = risk_factors.high_systolic_blood_pressure
    get_exposure_mock.side_effect = lambda *args, **kwargs: build_table(0.5, year_start, year_end)
    get_rr_mock.side_effect = lambda *args, **kwargs: build_table(1.01, year_start, year_end)
    get_paf_mock.side_effect = lambda *args, **kwargs: build_table(1, year_start, year_end)
    get_mf_mock.side_effect = lambda *args, **kwargs: 0

    class Distribution:
        def __init__(self, *_, **__):
            pass

        def setup(self, builder):
            data = build_table([130, 15], year_start, year_end, ['age', 'year', 'sex', 'mean', 'std'])
            self.parameters = builder.lookup(data)

        def ppf(self, propensity):
            params = self.parameters(propensity.index)
            return norm(loc=params['mean'], scale=params['std']).ppf(propensity)

    get_distribution_mock.side_effect = lambda *args, **kwargs: Distribution(args, kwargs)

    component = ContinuousRiskComponent(risk)

    simulation = setup_simulation([TestPopulation(), component], 100000, input_config=config)
    pop_view = simulation.population.get_view([risk.name+'_propensity'])

    pop_view.update(pd.Series(0.00001, index=simulation.population.population.index))
    pump_simulation(simulation, iterations=1)

    expected_value = norm(loc=130, scale=15).ppf(0.00001)
    assert np.allclose(simulation.population.population[risk.name+'_exposure'], expected_value)

    pop_view.update(pd.Series(0.5, index=simulation.population.population.index))
    pump_simulation(simulation, iterations=1)

    expected_value = 130
    assert np.allclose(simulation.population.population[risk.name+'_exposure'], expected_value)

    pop_view.update(pd.Series(0.99999, index=simulation.population.population.index))
    pump_simulation(simulation, iterations=1)

    expected_value = norm(loc=130, scale=15).ppf(0.99999)
    assert np.allclose(simulation.population.population[risk.name+'_exposure'], expected_value)


def test_correlated_propensity(load_risk_corr_mock, config):

    correlation_matrix = pd.DataFrame({
        'high_systolic_blood_pressure':           [1, 0.282213017344475, 0.110525231808424, 0.130475437755401, 0.237914389663941],
        'high_body_mass_index':                   [0.282213017344475, 1, 0.0928986519575119, -0.119147761153339, 0.212531763837137],
        'high_total_cholesterol':                 [0.110525231808424, 0.0928986519575119, 1, 0.175454370605231, 0.0476387962101613],
        'smoking_prevalence_approach':            [0.130475437755401, -0.119147761153339, 0.175454370605231, 1, 0.0770317213079334],
        'high_fasting_plasma_glucose_continuous': [0.237914389663941, 0.212531763837137, 0.0476387962101613, 0.0770317213079334, 1],
        'risk_factor':                ['high_systolic_blood_pressure', 'high_body_mass_index',
                                       'high_total_cholesterol', 'smoking_prevalence_approach',
                                       'high_fasting_plasma_glucose_continuous'],
        })
    correlation_matrix['age'] = 30
    correlation_matrix['sex'] = 'Male'
    load_risk_corr_mock.return_value = correlation_matrix

    pop = pd.DataFrame({'age': [30]*100000, 'sex': ['Male']*100000})

    propensities = []
    for risk in [
            risk_factors.high_systolic_blood_pressure,
            risk_factors.high_body_mass_index,
            risk_factors.high_total_cholesterol,
            risk_factors.smoking_prevalence_approach,
            risk_factors.high_fasting_plasma_glucose_continuous]:
        propensities.append(correlated_propensity_factory(config)(pop, risk))

    matrix = np.corrcoef(np.array(propensities))
    assert np.allclose(correlation_matrix[['high_systolic_blood_pressure', 'high_body_mass_index',
                                           'high_total_cholesterol', 'smoking_prevalence_approach',
                                           'high_fasting_plasma_glucose_continuous']].values, matrix, rtol=0.15)


def test_uncorrelated_propensity():
    pop = pd.DataFrame({'age': [30]*1000000, 'sex': ['Male']*1000000})
    propensities = []
    for risk in [
            risk_factors.high_systolic_blood_pressure,
            risk_factors.high_body_mass_index,
            risk_factors.high_total_cholesterol,
            risk_factors.smoking_prevalence_approach,
            risk_factors.high_fasting_plasma_glucose_continuous]:
        propensities.append(uncorrelated_propensity(pop, risk))

    propensities = np.array(propensities)
    assert propensities.min() >= 0
    assert propensities.max() <= 1
    hist, _ = np.histogram(propensities, 100, density=True)
    assert np.all(np.abs(hist - 1) < 0.01)


def _fill_in_correlation_matrix(risk_order):
    matrix_base = pd.DataFrame({
        risk_order[0]: [1, 0.282213017344475, 0.110525231808424, 0.130475437755401],
        risk_order[1]: [0.282213017344475, 1, 0.0928986519575119, -0.119147761153339],
        risk_order[2]: [0.110525231808424, 0.0928986519575119, 1, 0.175454370605231],
        risk_order[3]: [0.130475437755401, -0.119147761153339, 0.175454370605231, 1],
        'risk_factor': risk_order,
        })
    correlation_matrix = pd.DataFrame()

    for age in range(0,120, 5):
        for sex in ['Male', 'Female']:
            m = matrix_base.copy()
            m['age'] = age
            m['sex'] = sex
            correlation_matrix = correlation_matrix.append(m)

    return matrix_base, correlation_matrix


@pytest.mark.skip
@pytest.mark.slow
def test_correlated_exposures(load_rc_matrices_mock, config):
    from rpy2.robjects import r, pandas2ri, numpy2ri
    pandas2ri.activate()
    numpy2ri.activate()
    #config.population.pop_age_start = 30
    #config.population.pop_age_end = 100
    config.population.initial_age = 50
    draw = config.run_configuration.input_draw_number
    categorical_risks = [risk_factors.no_access_to_handwashing_facility, risk_factors.smoking_prevalence_approach]
    continuous_risks = [risk_factors.high_systolic_blood_pressure, risk_factors.high_total_cholesterol]

    categorical_risk_order = [r.name for r in categorical_risks]
    continuous_risk_order = [r.name for r in continuous_risks]
    risk_order = categorical_risk_order + continuous_risk_order

    matrix_base, load_rc_matrices_mock.return_value = _fill_in_correlation_matrix(risk_order)

    observations = []
    for i in range(100):
        print('running {}'.format(i))
        config.run_configuration.input_draw_number = i
        risks = [CategoricalRiskComponent(r, correlated_propensity_factory(draw)) for r in categorical_risks]
        risks += [ContinuousRiskComponent(r, correlated_propensity_factory(draw)) for r in continuous_risks]
        simulation = setup_simulation([TestPopulation()] + risks, 100000, input_config=config)

        pump_simulation(simulation, iterations=1)
        print('simulation done')

        r.source('/home/alecwd/Code/cost_effectiveness_misc/03_get_corr_matrix_function.R')
        pop = simulation.population.population[[c for c in simulation.population.population.columns if 'exposure' in c]]
        pop.columns = [c if 'exposure' not in c else c.rpartition('_')[0] for c in pop.columns]

        for risk in categorical_risks:
            pop[risk.name] = pop[risk.name].astype('category')

        pop = pop[risk_order]

        observations.append(pandas2ri.ri2py(r.get_corr_matrix(dat=pop, all_risks=r.c(*risk_order),
                                                              dichotomous_risks=r.c(*categorical_risk_order))))

    assert np.all(np.abs(matrix_base[risk_order].values-np.array(observations).mean(axis=0))
                  <= np.array(observations).std(axis=0)*3)


def inputs_mock_factory(config, input_type):
    year_start = config.simulation_parameters.year_start
    year_end = config.simulation_parameters.year_end

    def _mock_get_exposure_means(risk_id):
        e = {1: 0.5, 2: 0.25, 3: 0.1, 4: 0.8}[risk_id]
        if risk_id in [3, 4]:
            # Categorical risks
            return build_table([e, 1-e], year_start, year_end, columns=['age', 'year', 'sex', 'cat1', 'cat2'])
        else:
            return build_table(e, year_start, year_end)

    def _mock_get_relative_risk(risk_id, cause_id):
        e = {1: 0, 2: 0, 3: 0, 4: 0}[risk_id]
        if risk_id in [3, 4]:
            # Categorical risks
            return build_table([e, e], year_start, year_end, columns=['age', 'year', 'sex', 'cat1', 'cat2'])
        else:
            return build_table(e, year_start, year_end)

    def _mock_get_pafs(risk_id, cause_id):
        e = {1: 0, 2: 0, 3: 0, 4: 0}[risk_id]
        if risk_id in [3, 4]:
            # Categorical risks
            return build_table([e, e], year_start, year_end, columns=['age', 'year', 'sex', 'cat1', 'cat2'])
        else:
            return build_table(e, year_start, year_end)

    return {'exposure': _mock_get_exposure_means, 'rr': _mock_get_relative_risk, 'paf': _mock_get_pafs}[input_type]


@pytest.mark.skip
def test_correlated_exposures_synthetic_risks(load_risk_corr_mock, get_paf_mock, get_rr_mock,
                                              get_exposure_mock, config):
    from rpy2.robjects import r, pandas2ri, numpy2ri
    pandas2ri.activate()
    numpy2ri.activate()
    draw = config.run_configuration.input_draw_number

    load_risk_corr_mock.return_value = _fill_in_correlation_matrix()
    get_exposure_mock.side_effect = inputs_mock_factory(config, 'exposure')
    get_rr_mock.side_effect = inputs_mock_factory(config, 'rr')
    get_paf_mock.side_effect = inputs_mock_factory(config, 'paf')

    def loader(builder):
        dist = Interpolation(
                build_table([130, 15], ['age', 'year', 'sex', 'mean', 'std']),
                ['sex'],
                ['age', 'year'],
                func=lambda parameters: norm(loc=parameters['mean'], scale=parameters['std']).ppf)
        return builder.lookup(dist)

    continuous_1 = ConfigTree({'name': 'continuous_1', 'gbd_risk': 1, 'risk_type': 'continuous',
                               'affected_causes': [], 'tmrl': 112.5, 'scale': 10})
    continuous_2 = ConfigTree({'name': 'continuous_2', 'gbd_risk': 2, 'risk_type': 'continuous',
                               'affected_causes': [], 'tmrl': 3.08, 'scale': 1})
    categorical_1 = ConfigTree({'name': 'categorical_1',
                                'gbd_risk': 3,
                                'affected_causes': [],
                                'risk_type': 'categorical'})
    categorical_2 = ConfigTree({'name': 'categorical_2',
                                'gbd_risk': 4,
                                'affected_causes': [],
                                'risk_type': 'categorical'})

    distribution_map[continuous_1.name] = loader
    distribution_map[continuous_2.name] = loader
    distribution_map[continuous_1.name] = loader
    distribution_map[continuous_2.name] = loader

    continuous_1_component = ContinuousRiskComponent(continuous_1, correlated_propensity_factory(draw))
    continuous_2_component = ContinuousRiskComponent(continuous_2, correlated_propensity_factory(draw))
    categorical_1_component = CategoricalRiskComponent(categorical_1, correlated_propensity_factory(draw))
    categorical_2_component = CategoricalRiskComponent(categorical_2, correlated_propensity_factory(draw))
    components = [TestPopulation(), continuous_1_component, continuous_2_component,
                  categorical_1_component, categorical_2_component]
    simulation = setup_simulation(components, 10000, input_config=config)

    pump_simulation(simulation, iterations=1)

    r.source('/home/alecwd/Code/cost_effectiveness_misc/03_get_corr_matrix_function.R')
    pop = simulation.population.population[[c for c in simulation.population.population.columns if 'exposure' in c]]
    pop.columns = [c if 'exposure' not in c else c.rpartition('_')[0] for c in pop.columns]
    pop['categorical_1'] = pop.categorical_1.astype('category')
    pop['categorical_2'] = pop.categorical_2.astype('category')
    from rpy2.rinterface import RRuntimeError
    failure = None

    try:
        observed_correlation = pandas2ri.ri2py(
            r.get_corr_matrix(dat=pop, all_risks=r.c("continuous_1", "continuous_2", "categorical_1", "categorical_2"),
                              dichotomous_risks=r.c('categorical_1', 'categorical_2')))
    except Exception as e:
        print('test')
        print(e)
        print('\n'.join(r('unlist(traceback())')))
    assert np.allclose(matrix_base[["continuous_1", "continuous_2", "categorical_1", "categorical_2"]].values,
                       observed_correlation, rtol=0.25)


@pytest.mark.skip("Old and outdated test.  Needs serious rewrite.")
def test_make_gbd_risk_effects(config):
    time_step = config.simulation_parameters.time_step
    year_start = config.simulation_parameters.year_start
    year_end = config.simulation_parameters.year_end

    # adjusted pafs
    paf = 0.9
    mediation_factor = 0.02
    bmi_data_funcs = {
        'rr': lambda *args: build_table(0, year_start, year_end),
        'paf': lambda *args: build_table(paf, year_start, year_end),
        'mf': lambda *args: mediation_factor,
    }

    bmi = ContinuousRiskComponent(risk_factors.high_body_mass_index)

    simulation = setup_simulation([TestPopulation(), bmi], input_config=config)

    pafs = simulation.values.get_value('acute_hemorrhagic_stroke.paf', list_combiner, joint_value_post_processor)
    pafs.source = lambda index: [pd.Series(0, index=index)]
    assert np.allclose(pafs(simulation.population.population.index), paf * (1 - mediation_factor))

    # adjusted rrs
    rr = 1.26
    mediation_factor = 0.5
    adjrr = rr ** (1 - mediation_factor)
    tmrl = 21
    scale = 5
    exposure = 30

    bmi_data_funcs = {
        'rr': lambda *args: build_table(rr, year_start, year_end),
        'paf': lambda *args: build_table(0, year_start, year_end),
        'mf': lambda *args: mediation_factor,
    }
    bmi = ContinuousRiskComponent(risk_factors.high_body_mass_index)
    healthy = BaseDiseaseState('healthy')
    heart_attack = DiseaseState(sequelae.heart_attack)
    heart_attack_transition = RateTransition(healthy, heart_attack, get_data_functions={
        'incidence': lambda *args: build_table(.001, year_start, year_end)})
    simulation = setup_simulation([TestPopulation(), heart_attack_transition, bmi], input_config=config)
    irs = simulation.values.get_rate('heart_attack.incidence_rate')
    base_ir = irs.source(simulation.population.population.index)

    assert np.allclose(irs(simulation.population.population.index),
                       base_ir * max(adjrr**((exposure-tmrl)/scale), 1) * time_step/365, rtol=0.05)
