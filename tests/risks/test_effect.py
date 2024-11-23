from typing import Any

import numpy as np
import pandas as pd
import pytest
from layered_config_tree import LayeredConfigTree
from vivarium import Component, InteractiveContext
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.testing_utilities import TestPopulation

from vivarium_public_health.disease import SI
from vivarium_public_health.risks import RiskEffect
from vivarium_public_health.risks.base_risk import Risk

#
# from vivarium.framework.utilities import from_yearly
# from vivarium.testing_utilities import build_table, TestPopulation
# from vivarium.interface.interactive import initialize_simulation
#
# from vivarium_public_health.disease import RateTransition
from vivarium_public_health.risks.effect import NonLogLinearRiskEffect, RiskEffect
from vivarium_public_health.utilities import EntityString


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
def _setup_risk_effect_simulation(
    config: LayeredConfigTree,
    plugins: LayeredConfigTree,
    risk: str | Risk,
    risk_effect: RiskEffect,
    data: dict[str, Any],
) -> InteractiveContext:
    components = [
        TestPopulation(),
        risk,
        SI("test_cause"),
        risk_effect,
    ]

    simulation = InteractiveContext(
        components=components,
        configuration=config,
        plugin_configuration=plugins,
        setup=False,
    )

    for key, value in data.items():
        simulation._data.write(key, value)

    simulation.setup()
    return simulation


def build_dichotomous_risk_effect_data(rr_value: float) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "affected_entity": "test_cause",
            "affected_measure": "incidence_rate",
            "year_start": 1990,
            "year_end": 1991,
            "value": [rr_value, 1.0],
        },
        index=pd.Index(["cat1", "cat2"], name="parameter"),
    )
    return df


@pytest.mark.parametrize(
    "rr_source, rr_value",
    [("str", 2.0), ("float", 0.9), ("DataFrame", 0.5)],
)
def test_rr_sources(rr_source, rr_value, dichotomous_risk, base_config, base_plugins):
    risk = dichotomous_risk[0]
    effect = RiskEffect(risk.name, "cause.test_cause.incidence_rate")
    base_config.update({"risk_factor.test_risk": {"data_sources": {"exposure": 1.0}}})

    # TMREL of 1
    tmred = {"distribution": "uniform", "min": 1, "max": 1, "inverted": False}

    data = {
        f"{risk.name}.tmred": tmred,
        f"{risk.name}.population_attributable_fraction": 0,
        "cause.test_cause.incidence_rate": 1,
    }

    if rr_source == "DataFrame":
        rr_data = build_dichotomous_risk_effect_data(rr_value)
        base_config.update(
            {
                "risk_effect.test_risk_on_cause.test_cause.incidence_rate": {
                    "data_sources": {"relative_risk": rr_data}
                }
            }
        )
    elif rr_source == "float":
        base_config.update(
            {
                "risk_effect.test_risk_on_cause.test_cause.incidence_rate": {
                    "data_sources": {"relative_risk": rr_value}
                }
            }
        )
    else:  # rr_source is a string because it gets read from RiskEffect's configuration defaults
        rr_data = build_dichotomous_risk_effect_data(rr_value)
        data[f"{risk.name}.relative_risk"] = rr_data

    base_config.update({"risk_factor.test_risk": {"distribution_type": "dichotomous"}})
    simulation = _setup_risk_effect_simulation(base_config, base_plugins, risk, effect, data)

    pop = simulation.get_population()
    rate = simulation.get_value("test_cause.incidence_rate")(
        pop.index, skip_post_processor=True
    )
    assert set(rate.unique()) == {rr_value}


##############################
# Non Log-Linear Risk Effect #
##############################

custom_exposure_values = [0.5, 1, 1.5, 1.75, 2, 3, 4, 5, 5.5, 10]


class CustomExposureRisk(Component):
    """Risk where we define the exposure manually."""

    @property
    def name(self) -> str:
        return self.risk

    @property
    def columns_created(self) -> list[str]:
        return [self.exposure_column_name]

    def __init__(self, risk: str):
        super().__init__()
        self.risk = EntityString(risk)
        self.exposure_column_name = f"{self.risk.name}_exposure"

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        exposure_col = pd.Series(custom_exposure_values, name=self.exposure_column_name)
        self.population_view.update(exposure_col)

    def on_time_step_prepare(self, event: Event) -> None:
        exposure_col = pd.Series(custom_exposure_values, name=self.exposure_column_name)
        self.population_view.update(exposure_col)

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder):
        builder.value.register_value_producer(
            f"{self.risk.name}.exposure",
            source=self.get_exposure,
        )

    def get_exposure(self, index: pd.Index) -> pd.Series:
        data = pd.Series(custom_exposure_values, index=index)
        return data


@pytest.mark.parametrize(
    "rr_parameter_data, error_message",
    [
        ([1, 2, 5], None),
        ([2, 1, 5], "monotonic"),
        (["cat1", "cat2", "cat3"], "numeric"),
        (["per unit", "per unit", "per unit"], "numeric"),
    ],
)
def test_non_loglinear_effect(rr_parameter_data, error_message, base_config, base_plugins):
    risk = CustomExposureRisk("risk_factor.test_risk")
    effect = NonLogLinearRiskEffect(risk.name, "cause.test_cause.incidence_rate")

    risk_effect_rrs = [2.0, 2.4, 4.0]
    rr_data = pd.DataFrame(
        {
            "affected_entity": "test_cause",
            "affected_measure": "incidence_rate",
            "year_start": 1990,
            "year_end": 1991,
            "parameter": rr_parameter_data,
            "value": risk_effect_rrs,
        },
    )
    # enforce TMREL of 1
    tmred = {"distribution": "uniform", "min": 1, "max": 1, "inverted": False}

    data = {
        f"{risk.name}.relative_risk": rr_data,
        f"{risk.name}.tmred": tmred,
        f"{risk.name}.population_attributable_fraction": 0,
        "cause.test_cause.incidence_rate": 1,
    }

    base_config.update({"population": {"population_size": 10}})

    if error_message:
        with pytest.raises(ValueError, match=error_message):
            simulation = _setup_risk_effect_simulation(
                base_config, base_plugins, risk, effect, data
            )
        return
    else:
        simulation = _setup_risk_effect_simulation(
            base_config, base_plugins, risk, effect, data
        )

    pop = simulation.get_population()
    rate = simulation.get_value("test_cause.incidence_rate")(
        pop.index, skip_post_processor=True
    )
    expected_values = np.interp(
        custom_exposure_values,
        rr_parameter_data,
        np.array(risk_effect_rrs) / 2,  # RRs get divided by RR at TMREL
    )

    assert np.isclose(rate.values, expected_values, rtol=0.0000001).all()


def test_relative_risk_pipeline(dichotomous_risk, base_config, base_plugins):
    risk = dichotomous_risk[0]
    effect = RiskEffect(risk.name, "cause.test_cause.incidence_rate")
    base_config.update({"risk_factor.test_risk": {"data_sources": {"exposure": 0.75}}})

    # TMREL of 1
    tmred = {"distribution": "uniform", "min": 1, "max": 1, "inverted": False}

    data = {
        f"{risk.name}.tmred": tmred,
        f"{risk.name}.population_attributable_fraction": 0,
        "cause.test_cause.incidence_rate": 1,
    }
    rr_value = 1.4
    base_config.update(
        {
            "risk_effect.test_risk_on_cause.test_cause.incidence_rate": {
                "data_sources": {"relative_risk": rr_value}
            }
        }
    )

    base_config.update({"risk_factor.test_risk": {"distribution_type": "dichotomous"}})
    sim = _setup_risk_effect_simulation(base_config, base_plugins, risk, effect, data)
    pop = sim.get_population()

    expected_pipeline_name = f"{effect.risk.name}_on_{effect.target.name}.relative_risk"
    assert expected_pipeline_name in sim.list_values()

    rr_mapper = {
        "cat1": 1.4,
        "cat2": 1.0,
    }
    for exposure in rr_mapper:
        exposure_pipeline = sim.get_value(f"{effect.risk.name}.exposure")(pop.index)
        exposure_idx = exposure_pipeline.loc[exposure_pipeline == exposure].index
        relative_risk = sim.get_value(expected_pipeline_name)(exposure_idx)
        assert (relative_risk == rr_mapper[exposure]).all()
