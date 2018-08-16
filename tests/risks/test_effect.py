import pytest
import numpy as np
import pandas as pd
from scipy.stats import norm

from vivarium.framework.utilities import from_yearly
from vivarium.testing_utilities import build_table, TestPopulation, metadata
from vivarium.interface.interactive import setup_simulation, initialize_simulation

from vivarium_public_health.disease import RateTransition
from vivarium_public_health.risks.effect import continuous_exposure_effect, categorical_exposure_effect, RiskEffect
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


def test_RiskEffect(base_config, base_plugins):
    year_start = base_config.time.start.year
    year_end = base_config.time.end.year
    time_step = pd.Timedelta(days=base_config.time.step_size)
    test_exposure = [0]

    def test_function(rates_, rr):
        return rates_ * (rr.values**test_exposure[0])

    r = 'test_risk'
    d = 'test_cause'
    effect_data_functions = {
        'rr': lambda *args: build_table([1.01, 'per_unit'], year_start, year_end,
                                        ('age', 'year', 'sex', 'value', 'parameter')),
        'paf': lambda *args: build_table(0.01, year_start, year_end, ('age', 'year', 'sex', 'value')),
    }

    effect = RiskEffect(r, d, effect_data_functions)

    simulation = initialize_simulation([TestPopulation(), effect], input_config=base_config, plugin_config=base_plugins)
    simulation.data.write("risk_factor.test_risk.distribution", "dichotomuous")

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
        'rr': lambda *args: build_table([risk_rr, 'per_unit'], year_start, year_end,
                                        ('age', 'year', 'sex', 'value', 'parameter')),
        'paf': lambda *args: build_table(risk_paf, year_start, year_end, ('age', 'year', 'sex', 'value')),
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
    effect = RiskEffect('bad_risk', 'infected', effect_data_functions)

    rf_simulation = initialize_simulation([TestPopulation(), transition, effect],
                                          input_config=base_config, plugin_config=base_plugins)
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

    class exposure_function_wrapper:

        def setup(self, builder):
            self.population_view = builder.population.get_view([risk+'_exposure'])
            self.exposure_function = continuous_exposure_effect(risk, "risk_factor", self.population_view, builder)

        def __call__(self, *args, **kwargs):
            return self.exposure_function(*args, **kwargs)
    exposure_function = exposure_function_wrapper()

    tmrel = 0.5 * (tmred["max"] + tmred["min"])

    components = [TestPopulation(), make_dummy_column(risk+'_exposure', tmrel), exposure_function]
    simulation = initialize_simulation(components, input_config=base_config, plugin_config=base_plugins)
    simulation.data.write("risk_factor.test_risk.distribution", "ensemble")
    simulation.data.write("risk_factor.test_risk.tmred", tmred)
    simulation.data.write("risk_factor.test_risk.exposure_parameters", exposure_parameters)

    simulation.setup()

    rates = pd.Series(0.01, index=simulation.population.population.index)
    rr = pd.Series(1.01, index=simulation.population.population.index)

    assert np.all(exposure_function(rates, rr) == 0.01)

    simulation.population.get_view([risk+'_exposure']).update(
        pd.Series(tmrel + 50, index=simulation.population.population.index))

    expected_value = 0.01 * (1.01 ** (((tmrel + 50) - tmrel) / exposure_parameters["scale"]))

    assert np.allclose(exposure_function(rates, rr), expected_value)


def test_categorical_exposure_effect(base_config):
    risk = "test_risk"

    class exposure_function_wrapper:
        def setup(self, builder):
            self.population_view = builder.population.get_view([risk + '_exposure'])
            self.exposure_function = categorical_exposure_effect(risk, self.population_view)

        def __call__(self, *args, **kwargs):
            return self.exposure_function(*args, **kwargs)

    exposure_function = exposure_function_wrapper()
    components = [TestPopulation(), make_dummy_column(risk+'_exposure', 'cat2'), exposure_function]
    simulation = setup_simulation(components, input_config=base_config)

    rates = pd.Series(0.01, index=simulation.population.population.index)
    rr = pd.DataFrame({'cat1': 1.01, 'cat2': 1}, index=simulation.population.population.index)

    assert np.all(exposure_function(rates, rr) == 0.01)

    simulation.population.get_view([risk+'_exposure']).update(
        pd.Series('cat1', index=simulation.population.population.index))

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
    rr_data = build_table(
        [1.01, 1], year_start, year_end, ['age', 'year', 'sex', 'cat1', 'cat2']
    ).melt(id_vars=('age', 'year', 'sex'), var_name='parameter', value_name='value')

    simulation.data.write("risk_factor.test_risk.relative_risk", rr_data)
    simulation.data.write("risk_factor.test_risk.population_attributable_fraction", 1)
    affected_causes = ["test_cause_1", "test_cause_2"]
    simulation.data.write("risk_factor.test_risk.affected_causes", affected_causes)
    simulation.data.write("risk_factor.test_risk.distribution", "dichotomous")

    simulation.setup()

    simulation.step()

    incidence_rate = simulation.values.register_rate_producer(affected_causes[0]+'.incidence_rate')
    incidence_rate.source = simulation.tables.build_table(build_table(0.01, year_start, year_end),
                                                          key_columns=('sex',),
                                                          parameter_columns=('age', 'year'))

    assert np.isclose((simulation.population.population[risk+'_exposure'] == 'cat1').sum()
                      / len(simulation.population.population), 0.5, rtol=0.01)

    expected_exposed_value = 0.01 * 1.01
    expected_unexposed_value = 0.01

    exposed_index = simulation.population.population.index[
        simulation.population.population[risk+'_exposure'] == 'cat1']
    unexposed_index = simulation.population.population.index[
        simulation.population.population[risk+'_exposure'] == 'cat2']

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

    rr_data = build_table(
        [1.03, 1.02, 1.01, 1], year_start, year_end, ['age', 'year', 'sex', 'cat1', 'cat2', 'cat3', 'cat4']
    ).melt(id_vars=('age', 'year', 'sex'), var_name='parameter', value_name='value')

    affected_causes = ["test_cause_1", "test_cause_2"]
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

    for category in ['cat1', 'cat2', 'cat3', 'cat4']:
        assert np.isclose((simulation.population.population[risk+'_exposure'] == category).sum()
                          / len(simulation.population.population), 0.25, rtol=0.02)

    expected_exposed_value = 0.01 * np.array([1.02, 1.03, 1.01])

    for cat, expected in zip(['cat1', 'cat2', 'cat3', 'cat4'], expected_exposed_value):
        exposed_index = simulation.population.population.index[
            simulation.population.population[risk+'_exposure'] == cat]
        assert np.allclose(incidence_rate(exposed_index), from_yearly(expected, time_step), rtol=0.01)


def test_ContinuousRiskComponent(get_distribution_mock, base_config, base_plugins):
    time_step = pd.Timedelta(days=base_config.time.step_size)
    year_start = base_config.time.start.year
    year_end = base_config.time.end.year

    risk = "test_risk"

    exposure_data = build_table(
        0.5, year_start, year_end
    ).melt(id_vars=('age', 'year', 'sex'), var_name='parameter', value_name='value')

    rr_data = build_table(
        1.01, year_start, year_end
    ).melt(id_vars=('age', 'year', 'sex'), var_name='parameter', value_name='value')

    affected_causes = ["test_cause_1", "test_cause_2"]

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

    assert np.allclose(simulation.population.population[risk+'_exposure'], 130, rtol=0.001)

    expected_value = 0.01 * (1.01**((130 - 112) / 10))

    assert np.allclose(incidence_rate(simulation.population.population.index),
                       from_yearly(expected_value, time_step), rtol=0.001)
