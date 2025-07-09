from typing import Any

import numpy as np
import pandas as pd
from layered_config_tree import LayeredConfigTree
from vivarium import InteractiveContext
from vivarium.testing_utilities import TestPopulation

from vivarium_public_health.disease import SIS
from vivarium_public_health.risks.effect import InterventionEffect
from vivarium_public_health.risks.treatment.intervention import Intervention
from vivarium_public_health.utilities import EntityString


def _setup_intervention_simulation(
    config: LayeredConfigTree,
    plugins: LayeredConfigTree,
    intervention: str | Intervention,
    data: dict[str, Any],
    has_intervention_effect: bool = True,
) -> InteractiveContext:
    if isinstance(intervention, str):
        intervention = Intervention(intervention)
    components = [TestPopulation(), intervention]
    if has_intervention_effect:
        components.append(SIS("some_disease"))
        components.append(
            InterventionEffect(intervention.name, "cause.some_disease.incidence_rate")
        )

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


def _check_coverage_and_rr(
    simulation: InteractiveContext,
    intervention: EntityString,
    expected_coverage: dict[str, float],
    expected_rrs: dict[str, float],
) -> None:
    population = simulation.get_population()
    coverage = simulation.get_value(f"{intervention.name}.coverage")(population.index)
    incidence_rate = simulation.get_value("some_disease.incidence_rate")(population.index)
    unexposed_category = "covered"
    unexposed_incidence = incidence_rate[coverage == unexposed_category].iat[0]

    for category, category_coverage in expected_coverage.items():
        relative_risk = expected_rrs[category]
        is_in_category = coverage == category
        # todo use fuzzy checker for these tests
        assert np.isclose(is_in_category.mean(), category_coverage, 0.02)

        actual_incidence_rates = incidence_rate[is_in_category]
        expected_incidence_rates = unexposed_incidence * relative_risk
        assert np.isclose(actual_incidence_rates, expected_incidence_rates).all()


def test_dichotomous_intervention(base_config, base_plugins):
    intervention = Intervention("intervention.test_intervention")
    rr_data = pd.DataFrame(
        {
            "affected_entity": "some_disease",
            "affected_measure": "incidence_rate",
            "year_start": 1990,
            "year_end": 1991,
            "value": [1.5, 1.0],
        },
        index=pd.Index(["uncovered", "covered"], name="parameter"),
    )

    data = {
        f"{intervention.name}.coverage": pd.DataFrame(
            {
                "year_start": 1990,
                "year_end": 1991,
                "sex": ["Male"] * 2 + ["Female"] * 2,
                "parameter": ["uncovered", "covered"] * 2,
                "value": [0.25, 0.75] * 2,
            }
        ),
        f"{intervention.name}.relative_risk": rr_data.reset_index(),
        f"{intervention.name}.population_attributable_fraction": pd.DataFrame(
            {
                "affected_entity": "some_disease",
                "affected_measure": "incidence_rate",
                "year_start": 1990,
                "year_end": 1991,
                "value": 0.5,
            },
            index=[0],
        ),
        f"{intervention.name}.distribution_type": "dichotomous",
    }

    data_sources = {"data_sources": {"coverage": 0.25}}
    base_config.update(
        {
            "population": {"population_size": 50000},
            "risk_factor.test_risk": {
                **data_sources,
                **{"distribution_type": "dichotomous"},
            },
        }
    )
    category_exposures = {"uncovered": 0.25, "covered": 0.75}

    simulation = _setup_intervention_simulation(base_config, base_plugins, intervention, data)

    _check_coverage_and_rr(
        simulation, intervention.entity, category_exposures, rr_data["value"].to_dict()
    )

    simulation.step()

    _check_coverage_and_rr(
        simulation, intervention.entity, category_exposures, rr_data["value"].to_dict()
    )
