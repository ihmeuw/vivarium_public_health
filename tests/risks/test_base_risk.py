from typing import Any

import numpy as np
import pandas as pd
import pytest
from layered_config_tree import LayeredConfigTree
from vivarium import Component, InteractiveContext
from vivarium.framework.engine import Builder

from tests.test_utilities import build_table_with_age
from vivarium_public_health.disease import SIS
from vivarium_public_health.population import BasePopulation
from vivarium_public_health.risks import RiskEffect
from vivarium_public_health.risks.base_risk import ContinuousRisk, Risk
from vivarium_public_health.risks.distributions import (
    EnsembleDistribution,
    PolytomousDistribution,
)
from vivarium_public_health.utilities import EntityString


@pytest.fixture
def polytomous_risk() -> tuple[Risk, dict[str, Any]]:
    risk = "risk_factor.test_risk"
    risk_data = {}
    exposure_data = build_table_with_age(
        0.25, value_columns=["cat1", "cat2", "cat3", "cat4"]
    ).melt(
        id_vars=("age_start", "age_end", "year_start", "year_end", "sex"),
        var_name="parameter",
        value_name="value",
    )

    risk_data[f"{risk}.exposure"] = exposure_data
    risk_data[f"{risk}.categories"] = {
        "cat1": "severe",
        "cat2": "moderate",
        "cat3": "mild",
        "cat4": "unexposed",
    }
    risk_data[f"{risk}.distribution"] = "ordered_polytomous"
    risk_data[f"{risk}.relative_risk"] = pd.DataFrame(
        {
            "parameter": ["cat1", "cat2", "cat3", "cat4"],
            "affected_entity": "some_disease",
            "affected_measure": "incidence_rate",
            "year_start": 1990,
            "year_end": 1991,
            "value": [1.5, 1.2, 1.1, 1.0],
        }
    )
    risk_data[f"{risk}.population_attributable_fraction"] = pd.DataFrame(
        {
            "affected_entity": "some_disease",
            "affected_measure": "incidence_rate",
            "year_start": 1990,
            "year_end": 1991,
            "value": 0.5,
        },
        index=[0],
    )
    return Risk(risk), risk_data


def _setup_risk_simulation(
    config: LayeredConfigTree,
    plugins: LayeredConfigTree,
    risk: str | Risk,
    data: dict[str, Any],
    has_risk_effect: bool = True,
) -> InteractiveContext:
    if isinstance(risk, str):
        risk = Risk(risk)
    components = [BasePopulation(), risk]
    if has_risk_effect:
        components.append(SIS("some_disease"))
        components.append(RiskEffect(risk.name, "cause.some_disease.incidence_rate"))

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


# @pytest.mark.parametrize('propensity', [0.00001, 0.5, 0.99])
# def test_propensity_effect(propensity, mocker, continuous_risk, base_config, base_plugins):
#     population_size = 1000
#
#     rf, risk_data = continuous_risk
#     base_config.update({'population': {'population_size': population_size}}, **metadata(__file__))
#     sim = initialize_simulation([BasePopulation(), rf], input_config=base_config, plugin_config=base_plugins)
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
#     simulation = initialize_simulation([BasePopulation(), dummy_risk],
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


def test_polytomous_risk_lookup_configuration(polytomous_risk, base_config, base_plugins):
    risk, risk_data = polytomous_risk

    _setup_risk_simulation(base_config, base_plugins, risk, risk_data, has_risk_effect=False)

    # We have to get the distribution component's lookup tables. This is the distribution class
    # instantiated by the sub_component of the risk class

    assert isinstance(risk.exposure_distribution, PolytomousDistribution)


def _check_exposure_and_rr(
    simulation: InteractiveContext,
    risk: EntityString,
    expected_exposures: dict[str, float],
    expected_rrs: dict[str, float],
) -> None:
    population = simulation.get_population(
        [f"{risk.name}.exposure", "some_disease.incidence_rate"]
    )
    exposure = population[f"{risk.name}.exposure"]
    incidence_rate = population["some_disease.incidence_rate"]
    unexposed_category = sorted(expected_exposures.keys())[-1]
    unexposed_incidence = incidence_rate[exposure == unexposed_category].iat[0]

    for category, expected_exposure in expected_exposures.items():
        relative_risk = expected_rrs[category]
        is_in_category = exposure == category
        # todo use fuzzy checker for these tests
        assert np.isclose(is_in_category.mean(), expected_exposure, 0.02)

        actual_incidence_rates = incidence_rate[is_in_category]
        expected_incidence_rates = unexposed_incidence * relative_risk
        assert np.isclose(actual_incidence_rates, expected_incidence_rates).all()


def test_polytomous_risk(polytomous_risk, base_config, base_plugins):
    risk, risk_data = polytomous_risk
    rr_data = risk_data[f"{risk.name}.relative_risk"].set_index("parameter")
    exposure_data = risk_data[f"{risk.name}.exposure"].groupby("parameter")["value"].mean()

    base_config.update({"population": {"population_size": 50000}})

    simulation = _setup_risk_simulation(base_config, base_plugins, risk, risk_data)

    _check_exposure_and_rr(
        simulation,
        risk.risk,
        exposure_data.to_dict(),
        rr_data["value"].to_dict(),
    )

    simulation.step()

    _check_exposure_and_rr(
        simulation,
        risk.risk,
        exposure_data.to_dict(),
        rr_data["value"].to_dict(),
    )


@pytest.mark.parametrize("scalar_exposure", [True, False])
def test_dichotomous_risk(base_config, base_plugins, scalar_exposure):
    risk = Risk("risk_factor.test_risk")
    rr_data = pd.DataFrame(
        {
            "affected_entity": "some_disease",
            "affected_measure": "incidence_rate",
            "year_start": 1990,
            "year_end": 1991,
            "value": [1.5, 1.0],
        },
        index=pd.Index(["cat1", "cat2"], name="parameter"),
    )

    data = {
        f"{risk.name}.exposure": pd.DataFrame(
            {
                "year_start": 1990,
                "year_end": 1991,
                "sex": ["Male"] * 2 + ["Female"] * 2,
                "parameter": ["cat1", "cat2"] * 2,
                "value": [0.25, 0.75] * 2,
            }
        ),
        f"{risk.name}.relative_risk": rr_data.reset_index(),
        f"{risk.name}.population_attributable_fraction": pd.DataFrame(
            {
                "affected_entity": "some_disease",
                "affected_measure": "incidence_rate",
                "year_start": 1990,
                "year_end": 1991,
                "value": 0.5,
            },
            index=[0],
        ),
    }

    data_sources = {"data_sources": {"exposure": 0.25}} if scalar_exposure else {}
    base_config.update(
        {
            "population": {"population_size": 50000},
            "risk_factor.test_risk": {
                **data_sources,
                **{"distribution_type": "dichotomous"},
            },
        }
    )
    category_exposures = {"cat1": 0.25, "cat2": 0.75}

    simulation = _setup_risk_simulation(base_config, base_plugins, risk, data)

    _check_exposure_and_rr(
        simulation, risk.risk, category_exposures, rr_data["value"].to_dict()
    )

    simulation.step()

    _check_exposure_and_rr(
        simulation, risk.risk, category_exposures, rr_data["value"].to_dict()
    )


def test_ensemble_risk(base_config, base_plugins):
    risk = Risk("risk_factor.test_risk")

    distribution_weights = {
        "betasr": 0.055,
        "exp": 0.06,
        "gamma": 0.065,
        "glnorm": 0,
        "gumbel": 0.07,
        "invgamma": 0.075,
        "invweibull": 0.8,
        "llogis": 0.085,
        "lnorm": 0.09,
        "mgamma": 0.095,
        "mgumbel": 0.1,
        "norm": 0.105,
        "weibull": 0.12,
    }

    data = {
        f"{risk.name}.exposure": pd.DataFrame(
            {
                "year_start": 1990,
                "year_end": 1991,
                "parameter": "continuous",
                "value": 5.0,
            },
            index=[0],
        ),
        f"{risk.name}.exposure_standard_deviation": pd.DataFrame(
            {
                "year_start": 1990,
                "year_end": 1991,
                "value": 0.5,
            },
            index=[0],
        ),
        f"{risk.name}.exposure_distribution_weights": pd.DataFrame(
            {
                "year_start": 1990,
                "year_end": 1991,
                "parameter": list(distribution_weights.keys()),
                "value": list(distribution_weights.values()),
            },
        ),
        f"{risk.name}.population_attributable_fraction": pd.DataFrame(
            {
                "affected_entity": "some_disease",
                "affected_measure": "incidence_rate",
                "year_start": 1990,
                "year_end": 1991,
                "value": 0.5,
            },
            index=[0],
        ),
    }

    base_config.update(
        {
            "risk_factor.test_risk": {
                "data_sources": {"exposure": 0.25},
                "distribution_type": "ensemble",
                "ensemble_members": 2,
            },
            f"risk_effect.test_risk_on_some_disease.incidence_rate": {
                "distribution_args": {"relative_risk": 1.5}
            },
        }
    )

    simulation = _setup_risk_simulation(
        base_config, base_plugins, risk, data, has_risk_effect=False
    )

    # Get the distribution component
    distribution = risk.exposure_distribution

    assert isinstance(distribution, EnsembleDistribution)

    expected_distributions = set(distribution_weights.keys()) - {"glnorm"}
    assert expected_distributions == set(distribution.parameters.keys())

    simulation.step()

    # todo: use fuzzy checker to confirm that we are getting the expected results
    print("We didn't runtime error - success!")


def test_continuous_risk_matches_risk(base_config_factory, base_plugins):
    """Test that ContinuousRisk with default calibration_constant=0 produces
    the same exposures as Risk for a continuous (normal) distribution."""
    population_size = 10000

    exposure_data = pd.DataFrame(
        {
            "year_start": 1990,
            "year_end": 1991,
            "parameter": "continuous",
            "value": 130.0,
        },
        index=[0],
    )
    exposure_sd_data = pd.DataFrame(
        {
            "year_start": 1990,
            "year_end": 1991,
            "value": 15.0,
        },
        index=[0],
    )

    data = {
        "risk_factor.test_risk.exposure": exposure_data,
        "risk_factor.test_risk.exposure_standard_deviation": exposure_sd_data,
    }

    config_updates = {
        "population": {"population_size": population_size},
        "risk_factor.test_risk": {"distribution_type": "normal"},
    }

    # Create simulation with Risk
    config_risk = base_config_factory()
    config_risk.update(config_updates)
    risk = Risk("risk_factor.test_risk")
    sim_risk = _setup_risk_simulation(
        config_risk, base_plugins, risk, data, has_risk_effect=False
    )
    risk_exposure = sim_risk.get_population(["test_risk.exposure"])["test_risk.exposure"]

    # Create simulation with ContinuousRisk
    config_continuous = base_config_factory()
    config_continuous.update(config_updates)
    continuous_risk = ContinuousRisk("risk_factor.test_risk")
    sim_continuous = _setup_risk_simulation(
        config_continuous, base_plugins, continuous_risk, data, has_risk_effect=False
    )
    continuous_exposure = sim_continuous.get_population(["test_risk.exposure"])[
        "test_risk.exposure"
    ]

    pd.testing.assert_series_equal(risk_exposure, continuous_exposure)


class _CalibrationConstantModifier(Component):
    """Test helper that modifies the calibration constant pipeline to return a
    fixed value."""

    def __init__(self, risk: str, value: float):
        super().__init__()
        self._risk = risk
        self._value = value

    def setup(self, builder: Builder) -> None:
        builder.value.register_attribute_modifier(
            f"{self._risk}_calibration_constant",
            modifier=self.modifier,
        )

    def modifier(self, index: pd.Index) -> pd.Series:
        return pd.Series(self._value, index=index)


def test_continuous_risk_calibration_constant(base_config, base_plugins):
    """Test that when the calibration constant pipeline is modified to 0.75,
    ContinuousRisk exposures are scaled by (1 - 0.75) = 0.25."""
    population_size = 10000
    calibration_value = 0.75

    exposure_data = pd.DataFrame(
        {
            "year_start": 1990,
            "year_end": 1991,
            "parameter": "continuous",
            "value": 130.0,
        },
        index=[0],
    )
    exposure_sd_data = pd.DataFrame(
        {
            "year_start": 1990,
            "year_end": 1991,
            "value": 15.0,
        },
        index=[0],
    )

    data = {
        "risk_factor.test_risk.exposure": exposure_data,
        "risk_factor.test_risk.exposure_standard_deviation": exposure_sd_data,
    }

    config_updates = {
        "population": {"population_size": population_size},
        "risk_factor.test_risk": {"distribution_type": "normal"},
    }

    # Build a ContinuousRisk simulation with a non-zero calibration constant
    base_config.update(config_updates)
    continuous_risk = ContinuousRisk("risk_factor.test_risk")
    modifier = _CalibrationConstantModifier("risk_factor.test_risk", calibration_value)

    components = [BasePopulation(), continuous_risk, modifier]
    simulation = InteractiveContext(
        components=components,
        configuration=base_config,
        plugin_configuration=base_plugins,
        setup=False,
    )
    for key, value in data.items():
        simulation._data.write(key, value)
    simulation.setup()

    # Get the raw ppf exposure (before calibration scaling) and the final exposure
    raw_exposure = simulation.get_population(
        [continuous_risk.exposure_distribution.exposure_ppf_pipeline]
    )[continuous_risk.exposure_distribution.exposure_ppf_pipeline]
    final_exposure = simulation.get_population(["test_risk.exposure"])["test_risk.exposure"]

    expected_exposure = raw_exposure * (1 - calibration_value)
    pd.testing.assert_series_equal(final_exposure, expected_exposure, check_names=False)
