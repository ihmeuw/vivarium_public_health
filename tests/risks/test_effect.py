import pytest
import numpy as np
import pandas as pd
from scipy.stats import norm

from vivarium.framework.utilities import from_yearly
from vivarium.testing_utilities import build_table, TestPopulation, metadata
from vivarium.interface.interactive import setup_simulation, initialize_simulation

from vivarium_public_health.disease import RateTransition
from vivarium_public_health.risks.effect import DirectEffect, get_exposure_effect
from vivarium_public_health.risks.base_risk import Risk


@pytest.fixture
def get_distribution_mock(mocker):
    return mocker.patch('vivarium_public_health.risks.base_risk.get_distribution')


def make_dummy_column(name, initial_value):
    class _make_dummy_column:
        def setup(self, builder):
            self.population_view = builder.population.get_view([name])
            builder.population.initializes_simulants(self.make_column, creates_columns=[name])

        def make_column(self, pop_data):
            self.population_view.update(pd.Series(initial_value, index=pop_data.index, name=name))
    return _make_dummy_column()


def test_RiskEffect(base_config, base_plugins, mocker):
    year_start = base_config.time.start.year
    year_end = base_config.time.end.year
    time_step = pd.Timedelta(days=base_config.time.step_size)
    test_exposure = [0]

    def test_function(rates_, rr):
        return rates_ * (rr.values**test_exposure[0])

    r = 'test_risk'
    d = 'test_cause'
    rf = Risk('risk_factor', r)
    effect_data_functions = {
        'rr': lambda *args: build_table([1.01, 'per_unit', d], year_start, year_end,
                                        ('age', 'year', 'sex', 'value', 'parameter', 'cause')),
        'paf': lambda *args: build_table([0.01, d], year_start, year_end, ('age', 'year', 'sex', 'value', 'cause')),
    }

    effect = DirectEffect(r, d, 'risk_factor', 'cause', effect_data_functions)

    simulation = initialize_simulation([TestPopulation(), effect], input_config=base_config, plugin_config=base_plugins)

    simulation.data.write("risk_factor.test_risk.distribution", "dichotomuous")
    simulation.values.register_value_producer("test_risk.exposure", mocker.Mock())

    simulation.setup()

    effect.exposure_effect = test_function

    # This one should be affected by our RiskEffect
    rates = simulation.values.register_rate_producer('test_cause.incidence_rate')
    rates.source = simulation.tables.build_table(build_table(0.01, year_start, year_end),
                                                 key_columns=('sex',),
                                                 parameter_columns=('age', 'year'))

    # This one should not
    other_rates = simulation.values.register_rate_producer('some_other_cause.incidence_rate')
    other_rates.source = simulation.tables.build_table(build_table(0.01, year_start, year_end),
                                                       key_columns=('sex',),
                                                       parameter_columns=('age', 'year'))

    assert np.allclose(rates(simulation.population.population.index), from_yearly(0.01, time_step))
    assert np.allclose(other_rates(simulation.population.population.index), from_yearly(0.01, time_step))

    test_exposure[0] = 1

    assert np.allclose(rates(simulation.population.population.index), from_yearly(0.0101, time_step))
    assert np.allclose(other_rates(simulation.population.population.index), from_yearly(0.01, time_step))


def test_risk_deletion(base_config, base_plugins, mocker):
    year_start = base_config.time.start.year
    year_end = base_config.time.end.year
    time_step = pd.Timedelta(days=base_config.time.step_size)

    base_rate = 0.01
    risk_paf = 0.5
    risk_rr = 1

    rate_data_functions = {
        'incidence_rate': lambda *args: build_table(0.01, year_start, year_end, ('age', 'year', 'sex', 'value'))
    }

    effect_data_functions = {
        'rr': lambda *args: build_table([risk_rr, 'per_unit', 'infected'], year_start, year_end,
                                        ('age', 'year', 'sex', 'value', 'parameter', 'cause')),
        'paf': lambda *args: build_table([risk_paf, 'infected'], year_start, year_end,
                                         ('age', 'year', 'sex', 'value', 'cause')),
    }

    def effect_function(rates, _):
        return rates

    transition = RateTransition(mocker.MagicMock(state_id='susceptible'),
                                mocker.MagicMock(state_id='infected'), rate_data_functions)

    base_simulation = initialize_simulation([TestPopulation(), transition],
                                            input_config=base_config, plugin_config=base_plugins)
    base_simulation.setup()

    incidence = base_simulation.get_value('infected.incidence_rate')
    joint_paf = base_simulation.get_value('infected.paf')

    # Validate the base case
    assert np.allclose(incidence(base_simulation.population.population.index), from_yearly(base_rate, time_step))
    assert np.allclose(joint_paf(base_simulation.population.population.index), 0)

    transition = RateTransition(mocker.MagicMock(state_id='susceptible'),
                                mocker.MagicMock(state_id='infected'), rate_data_functions)
    effect = DirectEffect('bad_risk', 'infected', 'risk_factor', 'cause', effect_data_functions)

    rf_simulation = initialize_simulation([TestPopulation(), transition, effect],
                                          input_config=base_config, plugin_config=base_plugins)

    rf_simulation.data.write("risk_factor.bad_risk.distribution", "dichotomuous")
    rf_simulation.values.register_value_producer("bad_risk.exposure", mocker.Mock())

    rf_simulation.setup()
    effect.exposure_effect = effect_function

    incidence = rf_simulation.get_value('infected.incidence_rate')
    joint_paf = rf_simulation.get_value('infected.paf')

    assert np.allclose(incidence(rf_simulation.population.population.index),
                       from_yearly(base_rate * (1 - risk_paf), time_step))
    assert np.allclose(joint_paf(rf_simulation.population.population.index), risk_paf)


def test_continuous_exposure_effect(base_config, base_plugins):
    risk = "test_risk"
    tmred = {
            "distribution": 'uniform',
            "min": 110.0,
            "max": 115.0,
            "inverted": False,
    }
    exposure_parameters = {
            "scale": 10.0,
            "max_rr": 200.0,
            "max_val": 300.0,
            "min_val": 50.0,
    }

    tmrel = 0.5 * (tmred["max"] + tmred["min"])

    class exposure_function_wrapper:

        def setup(self, builder):
            self.exposure_function = get_exposure_effect(builder, risk, 'risk_factor')

        def __call__(self, *args, **kwargs):
            return self.exposure_function(*args, **kwargs)

    exposure_function = exposure_function_wrapper()

    tmrel = 0.5 * (tmred["max"] + tmred["min"])

    components = [TestPopulation(), exposure_function]
    simulation = initialize_simulation(components, input_config=base_config, plugin_config=base_plugins)
    simulation.data.write("risk_factor.test_risk.distribution", "ensemble")
    simulation.data.write("risk_factor.test_risk.tmred", tmred)
    simulation.data.write("risk_factor.test_risk.exposure_parameters", exposure_parameters)

    def test_risk_exposure(index):
        return pd.Series(tmrel, index=index)

    simulation.values.register_value_producer('test_risk.exposure', source=test_risk_exposure)

    simulation.setup()

    rates = pd.Series(0.01, index=simulation.population.population.index)
    rr = pd.Series(1.01, index=simulation.population.population.index)

    assert np.all(exposure_function(rates, rr) == 0.01)

    def test_risk_exposure(index):
        return pd.Series(tmrel + 50, index=index)

    components = [TestPopulation(), exposure_function]
    simulation = initialize_simulation(components, input_config=base_config, plugin_config=base_plugins)
    simulation.data.write("risk_factor.test_risk.distribution", "ensemble")
    simulation.data.write("risk_factor.test_risk.tmred", tmred)
    simulation.data.write("risk_factor.test_risk.exposure_parameters", exposure_parameters)

    simulation.values.register_value_producer('test_risk.exposure', source=test_risk_exposure)

    simulation.setup()

    rates = pd.Series(0.01, index=simulation.population.population.index)
    rr = pd.Series(1.01, index=simulation.population.population.index)

    expected_value = 0.01 * (1.01 ** (((tmrel + 50) - tmrel) / exposure_parameters["scale"]))

    assert np.allclose(exposure_function(rates, rr), expected_value)


def test_categorical_exposure_effect(base_config, base_plugins, mocker):
    risk = "test_risk"
    risk_effect = mocker.Mock()
    risk_effect.risk = risk

    class exposure_function_wrapper:
        def setup(self, builder):
            self.exposure_function = get_exposure_effect(builder, risk, 'risk_factor')

        def __call__(self, *args, **kwargs):
            return self.exposure_function(*args, **kwargs)

    exposure_function = exposure_function_wrapper()
    components = [TestPopulation(), exposure_function]

    simulation = initialize_simulation(components, input_config=base_config, plugin_config=base_plugins)

    def test_risk_exposure(index):
        return pd.Series(['cat2'] * len(index), index=index)
    simulation.values.register_value_producer('test_risk.exposure', test_risk_exposure)
    simulation.data.write("risk_factor.test_risk.distribution", "dichotomous")
    simulation.setup()

    rates = pd.Series(0.01, index=simulation.population.population.index)
    rr = pd.DataFrame({'cat1': 1.01, 'cat2': 1}, index=simulation.population.population.index)

    assert np.all(exposure_function(rates, rr) == 0.01)

    simulation = initialize_simulation(components, input_config=base_config, plugin_config=base_plugins)

    def test_risk_exposure(index):
        return pd.Series(['cat1'] * len(index), index=index)

    simulation.values.register_value_producer('test_risk.exposure', test_risk_exposure)
    simulation.data.write("risk_factor.test_risk.distribution", "dichotomous")
    simulation.setup()

    rates = pd.Series(0.01, index=simulation.population.population.index)
    rr = pd.DataFrame({'cat1': 1.01, 'cat2': 1}, index=simulation.population.population.index)

    assert np.allclose(exposure_function(rates, rr), 0.0101)


def test_CategoricalRiskComponent_dichotomous_case(base_config, base_plugins):
    year_start = base_config.time.start.year
    year_end = base_config.time.end.year
    time_step = pd.Timedelta(days=base_config.time.step_size)
    base_config.update({'input_data': {'input_draw_number': 1}}, **metadata(__file__))
    risk = "test_risk"

    component = Risk("risk_factor", risk)
    base_config.update({'population': {'population_size': 100000}}, layer='override')
    simulation = initialize_simulation([TestPopulation(), component],
                                       input_config=base_config, plugin_config=base_plugins)

    exposure_data = build_table(
        0.5, year_start, year_end, ['age', 'year', 'sex', 'cat1', 'cat2']
    ).melt(id_vars=('age', 'year', 'sex'), var_name='parameter', value_name='value')

    simulation.data.write("risk_factor.test_risk.exposure", exposure_data)

    affected_causes = ["test_cause_1", "test_cause_2"]
    rr_data = []
    for cause in affected_causes:
        rr_data.append(
            build_table(
                [1.01, 1, cause], year_start, year_end, ['age', 'year', 'sex', 'cat1', 'cat2', 'cause']
            ).melt(id_vars=('age', 'year', 'sex', 'cause'), var_name='parameter', value_name='value')
        )
    rr_data = pd.concat(rr_data)

    simulation.data.write("risk_factor.test_risk.relative_risk", rr_data)
    simulation.data.write("risk_factor.test_risk.population_attributable_fraction", 1)

    simulation.data.write("risk_factor.test_risk.affected_causes", affected_causes)
    simulation.data.write("risk_factor.test_risk.distribution", "dichotomous")

    simulation.setup()

    simulation.step()

    incidence_rate = simulation.values.register_rate_producer(affected_causes[0]+'.incidence_rate')
    incidence_rate.source = simulation.tables.build_table(build_table(0.01, year_start, year_end),
                                                          key_columns=('sex',),
                                                          parameter_columns=('age', 'year'))

    categories = simulation.values.get_value('test_risk.exposure')(simulation.population.population.index)
    assert np.isclose(categories.value_counts()['cat1'] / len(simulation.population.population), 0.5, rtol=0.01)

    expected_exposed_value = 0.01 * 1.01
    expected_unexposed_value = 0.01

    exposed_index = categories[categories == 'cat1'].index
    unexposed_index = categories[categories == 'cat2'].index

    assert np.allclose(incidence_rate(exposed_index), from_yearly(expected_exposed_value, time_step))
    assert np.allclose(incidence_rate(unexposed_index), from_yearly(expected_unexposed_value, time_step))


def test_CategoricalRiskComponent_polytomous_case(base_config, base_plugins):
    year_start = base_config.time.start.year
    year_end = base_config.time.end.year
    time_step = pd.Timedelta(days=base_config.time.step_size)

    risk = "test_risk"

    component = Risk("risk_factor", risk)
    base_config.update({'population': {'population_size': 100000}}, layer='override')
    simulation = initialize_simulation([TestPopulation(), component],
                                       input_config=base_config, plugin_config=base_plugins)

    exposure_data = build_table(
        0.25, year_start, year_end, ['age', 'year', 'sex', 'cat1', 'cat2', 'cat3', 'cat4']
    ).melt(id_vars=('age', 'year', 'sex'), var_name='parameter', value_name='value')

    affected_causes = ["test_cause_1", "test_cause_2"]
    rr_data = []
    for cause in affected_causes:
        rr_data.append(
            build_table([1.03, 1.02, 1.01, 1, cause], year_start, year_end,
                        ['age', 'year', 'sex', 'cat1', 'cat2', 'cat3', 'cat4', 'cause']
                        ).melt(id_vars=('age', 'year', 'sex', 'cause'), var_name='parameter', value_name='value')
        )
    rr_data = pd.concat(rr_data)

    simulation.data.write("risk_factor.test_risk.exposure", exposure_data)
    simulation.data.write("risk_factor.test_risk.relative_risk", rr_data)
    simulation.data.write("risk_factor.test_risk.population_attributable_fraction", 1)
    simulation.data.write("risk_factor.test_risk.affected_causes", affected_causes)
    simulation.data.write("risk_factor.test_risk.distribution", "polytomous")
    simulation.setup()

    simulation.step()

    incidence_rate = simulation.values.register_rate_producer(affected_causes[0]+'.incidence_rate')
    incidence_rate.source = simulation.tables.build_table(build_table(0.01, year_start, year_end),
                                                          key_columns=('sex',),
                                                          parameter_columns=('age', 'year'))

    categories = simulation.values.get_value('test_risk.exposure')(simulation.population.population.index)

    for category in ['cat1', 'cat2', 'cat3', 'cat4']:
        assert np.isclose(categories.value_counts()[category] / len(simulation.population.population), 0.25, rtol=0.02)

    expected_exposed_value = 0.01 * np.array([1.02, 1.03, 1.01])

    for cat, expected in zip(['cat1', 'cat2', 'cat3', 'cat4'], expected_exposed_value):
        exposed_index = categories[categories == cat].index
        assert np.allclose(incidence_rate(exposed_index), from_yearly(expected, time_step), rtol=0.01)


def test_ContinuousRiskComponent(get_distribution_mock, base_config, base_plugins):
    time_step = pd.Timedelta(days=base_config.time.step_size)
    year_start = base_config.time.start.year
    year_end = base_config.time.end.year

    risk = "test_risk"

    exposure_data = build_table(
        0.5, year_start, year_end
    ).melt(id_vars=('age', 'year', 'sex'), var_name='parameter', value_name='value')

    affected_causes = ["test_cause_1", "test_cause_2"]

    rr_data = []
    for cause in affected_causes:
        rr_data.append(
            build_table([1.01, cause], year_start, year_end, ['age', 'sex', 'year', 'value', 'cause'],
                        ).melt(id_vars=('age', 'year', 'sex', 'cause'), var_name='parameter', value_name='value')
        )
    rr_data = pd.concat(rr_data)

    tmred = {
            "distribution": 'uniform',
            "min": 110.0,
            "max": 115.0,
            "inverted": False,
    }
    exposure_parameters = {
            "scale": 10.0,
            "max_rr": 200.0,
            "max_val": 300.0,
            "min_val": 50.0,
    }

    class Distribution:
        def __init__(self, *_, **__):
            pass

        def setup(self, builder):
            data = build_table([130, 0.000001], year_start, year_end, ['age', 'year', 'sex', 'mean', 'std'])
            self.parameters = builder.lookup.build_table(data)

        def ppf(self, propensity):
            params = self.parameters(propensity.index)
            return norm(loc=params['mean'], scale=params['std']).ppf(propensity)

    get_distribution_mock.side_effect = lambda *args, **kwargs: Distribution(args, kwargs)

    component = Risk("risk_factor", risk)

    base_config.update({'population': {'population_size': 100000}}, layer='override')
    simulation = initialize_simulation([TestPopulation(), component],
                                       input_config=base_config, plugin_config=base_plugins)
    simulation.data.write("risk_factor.test_risk.exposure", exposure_data)
    simulation.data.write("risk_factor.test_risk.relative_risk", rr_data)
    simulation.data.write("risk_factor.test_risk.population_attributable_fraction", 1)
    simulation.data.write("risk_factor.test_risk.affected_causes", affected_causes)
    simulation.data.write("risk_factor.test_risk.distribution", "ensemble")
    simulation.data.write("risk_factor.test_risk.tmred", tmred)
    simulation.data.write("risk_factor.test_risk.exposure_parameters", exposure_parameters)
    simulation.setup()

    simulation.step()

    incidence_rate = simulation.values.register_rate_producer(affected_causes[0]+'.incidence_rate')
    incidence_rate.source = simulation.tables.build_table(build_table(0.01, year_start, year_end),
                                                          key_columns=('sex',),
                                                          parameter_columns=('age', 'year'))

    exposure = simulation.values.get_value('test_risk.exposure')
    assert np.allclose(exposure(simulation.population.population.index), 130, rtol=0.001)

    expected_value = 0.01 * (1.01**((130 - 112) / 10))

    assert np.allclose(incidence_rate(simulation.population.population.index),
                       from_yearly(expected_value, time_step), rtol=0.001)


def test_IndirectEffect_dichotomous(base_config, base_plugins):
    year_start = base_config.time.start.year
    year_end = base_config.time.end.year
    base_config.update({'population': {'population_size': 100000}}, layer='override')
    affected_risk = Risk('risk_factor', 'test_risk')
    rf_exposed = 0.4

    rf_exposure_data = build_table(
        [rf_exposed, 1-rf_exposed], year_start, year_end, ['age', 'year', 'sex', 'cat1', 'cat2']
    ).melt(id_vars=('age', 'year', 'sex'), var_name='parameter', value_name='value')

    # start with the only risk factor without indirect effect from coverage_gap
    simulation = initialize_simulation([TestPopulation(), affected_risk],
                                       input_config=base_config, plugin_config=base_plugins)

    simulation.data.write("risk_factor.test_risk.exposure", rf_exposure_data)
    simulation.data.write("risk_factor.test_risk.distribution", "dichotomous")
    simulation.data.write("risk_factor.test_risk.affected_causes", [])

    simulation.setup()

    pop = simulation.population.population
    exposure = simulation.values.get_value('test_risk.exposure')
    assert np.isclose(rf_exposed, exposure(pop.index).value_counts()['cat1']/len(pop), rtol=0.01)

    # add the coverage gap which should change the exposure of test risk
    coverage_gap = Risk('coverage_gap', 'test_coverage_gap')
    simulation = initialize_simulation([TestPopulation(), affected_risk, coverage_gap],
                                       input_config=base_config, plugin_config=base_plugins)

    cg_exposed = 0.6
    cg_exposure_data = build_table(
        [cg_exposed, 1-cg_exposed], year_start, year_end, ['age', 'year', 'sex', 'cat1', 'cat2']
    ).melt(id_vars=('age', 'year', 'sex'), var_name='parameter', value_name='value')

    rr = 2
    rr_data = build_table(
        [rr, 1], year_start, year_end, ['age', 'year', 'sex', 'cat1', 'cat2']
    ).melt(id_vars=('age', 'year', 'sex'), var_name='parameter', value_name='value')

    rr_data['risk_factor'] = 'test_risk'

    # paf is (sum(exposure(category)*rr(category) -1 )/ (sum(exposure(category)* rr(category)
    paf = (rr * cg_exposed + (1-cg_exposed) - 1) / (rr * cg_exposed + (1-cg_exposed))

    paf_data = build_table(
        paf, year_start, year_end, ['age', 'year', 'sex', 'population_attributable_fraction']
    ).melt(id_vars=('age', 'year', 'sex'), var_name='population_attributable_fraction', value_name='value')

    paf_data['risk_factor'] = 'test_risk'

    simulation.data.write("risk_factor.test_risk.exposure", rf_exposure_data)
    simulation.data.write("risk_factor.test_risk.distribution", "dichotomous")
    simulation.data.write("risk_factor.test_risk.affected_causes", [])
    simulation.data.write("coverage_gap.test_coverage_gap.exposure", cg_exposure_data)
    simulation.data.write("coverage_gap.test_coverage_gap.distribution", "dichotomous")
    simulation.data.write("coverage_gap.test_coverage_gap.relative_risk", rr_data)
    simulation.data.write("coverage_gap.test_coverage_gap.affected_risk_factors", ['test_risk'])
    simulation.data.write("coverage_gap.test_coverage_gap.affected_causes", [])
    simulation.data.write("coverage_gap.test_coverage_gap.population_attributable_fraction", paf_data)

    simulation.setup()

    pop = simulation.population.population
    rf_exposure = simulation.values.get_value('test_risk.exposure')(pop.index)

    # proportion of simulants exposed to each category of affected risk stays same
    assert np.isclose(rf_exposed, rf_exposure.value_counts()['cat1']/len(pop), rtol=0.01)

    # compute relative risk to test whether it matches with the given relative risk
    cg_exposure = simulation.values.get_value('test_coverage_gap.exposure')(pop.index)

    cg_exposed = cg_exposure == 'cat1'
    rf_exposed = rf_exposure == 'cat1'

    affected_by_cg = rf_exposed & cg_exposed
    not_affected_by_cg = rf_exposed & ~cg_exposed

    computed_rr = (len(pop[affected_by_cg])/len(pop[cg_exposed])) / (len(pop[not_affected_by_cg])/len(pop[~cg_exposed]))
    assert np.isclose(computed_rr, rr, rtol=0.01)
