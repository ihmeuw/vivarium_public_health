import numpy as np
import pandas as pd
import pytest
from vivarium import Component, InteractiveContext
from vivarium.framework.lookup.table import InterpolatedTable

from vivarium_public_health.population import BasePopulation, Mortality
from vivarium_public_health.testing.mock_artifact import MockArtifact


def test_mortality_configuration(make_full_simulants, base_plugins, generate_population_mock):
    sims = make_full_simulants
    start_population_size = len(sims)

    generate_population_mock.return_value = sims.drop(columns=["tracked"])
    bp = BasePopulation()
    mortality = Mortality()

    sim = InteractiveContext(
        components=[bp, mortality], plugin_configuration=base_plugins, setup=False
    )
    override_config = {
        "population": {
            "population_size": start_population_size,
            "include_sex": "Male",
        },
        "mortality": {
            "lookup_tables": {
                "all_cause_mortality_rate": Component.build_lookup_table_config(
                    value=0.25,
                ),
            },
        },
    }
    sim.configuration.update(override_config)
    sim.setup()
    lookup_tables = mortality.lookup_tables
    expected_lookup_table_keys = [
        "all_cause_mortality_rate",
        "life_expectancy",
        "unmodeled_cause_specific_mortality_rate",
    ]

    assert set(expected_lookup_table_keys) == set(lookup_tables.keys())
    assert lookup_tables["all_cause_mortality_rate"].data == 0.25
    assert lookup_tables["unmodeled_cause_specific_mortality_rate"].data == 0.0
    assert isinstance(lookup_tables["life_expectancy"], InterpolatedTable)
