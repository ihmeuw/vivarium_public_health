from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd
import pytest
from layered_config_tree import LayeredConfigTree
from vivarium import InteractiveContext
from vivarium.framework.lookup.table import InterpolatedTable
from vivarium.testing_utilities import TestPopulation

from tests.test_utilities import build_table_with_age
from vivarium_public_health.disease import SIS
from vivarium_public_health.risks import RiskEffect
from vivarium_public_health.risks.base_risk import Risk
from vivarium_public_health.risks.distributions import (
    EnsembleDistribution,
    PolytomousDistribution,
)
from vivarium_public_health.utilities import EntityString


@pytest.fixture
def polytomous_risk() -> Tuple[Risk, Dict[str, Any]]:
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
    risk: Union[str, Risk],
    data: Dict[str, Any],
    has_risk_effect: bool = True,
) -> InteractiveContext:
    if isinstance(risk, str):
        risk = Risk(risk)
    components = [TestPopulation(), risk]
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


def test_polytomous_risk_lookup_configuration(polytomous_risk, base_config, base_plugins):
    risk, risk_data = polytomous_risk

    _setup_risk_simulation(base_config, base_plugins, risk, risk_data, has_risk_effect=False)

    # We have to get the distribution component's lookup tables. This is the distribution class
    # instantiated by the sub_component of the risk class

    assert isinstance(risk.exposure_distribution, PolytomousDistribution)

    lookup_tables = risk.exposure_distribution.lookup_tables

    # This risk is a PolytomousDistribution so there will only be an exposure lookup table
    assert {"exposure"} == set(lookup_tables.keys())
    assert isinstance(lookup_tables["exposure"], InterpolatedTable)


def _check_exposure_and_rr(
    simulation: InteractiveContext,
    risk: EntityString,
    expected_exposures: Dict[str, float],
    expected_rrs: Dict[str, float],
) -> None:
    population = simulation.get_population()
    exposure = simulation.get_value(f"{risk.name}.exposure")(population.index)
    incidence_rate = simulation.get_value("some_disease.incidence_rate")(population.index)
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
    assert {"ensemble_distribution_weights"} == set(distribution.lookup_tables.keys())
    assert expected_distributions == set(distribution.parameters.keys())

    simulation.step()

    # todo: use fuzzy checker to confirm that we are getting the expected results
    print("We didn't runtime error - success!")
