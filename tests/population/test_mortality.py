from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from vivarium import Component, InteractiveContext
from vivarium.framework.state_machine import Transition
from vivarium.testing_utilities import build_table
from vivarium_testing_utils import FuzzyChecker

from vivarium_public_health.disease import BaseDiseaseState, DiseaseModel, DiseaseState
from vivarium_public_health.population import BasePopulation, Mortality


@pytest.fixture
def setup_sim_with_pop_and_mortality(
    full_simulants, base_plugins, generate_population_mock
) -> Tuple[InteractiveContext, Component, Component]:

    # Initializes an Interactive context with BasePopulation and Mortality components
    start_population_size = len(full_simulants)

    generate_population_mock.return_value = full_simulants.drop(columns=["tracked"])
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
        # todo test with actual unmodeled causes and non-zero mortality rates
        "mortality": {"unmodeled_causes": []},
    }
    sim.configuration.update(override_config)
    sim.setup()
    return sim, bp, mortality


def test_mortality_default_lookup_configuration(setup_sim_with_pop_and_mortality):
    sim, bp, mortality = setup_sim_with_pop_and_mortality
    lookup_tables = mortality.lookup_tables
    expected_lookup_table_keys = [
        "all_cause_mortality_rate",
        "life_expectancy",
        "unmodeled_cause_specific_mortality_rate",
    ]

    assert set(expected_lookup_table_keys) == set(lookup_tables.keys())
    assert (lookup_tables["all_cause_mortality_rate"].data["value"] == 0.5).all()
    assert lookup_tables["unmodeled_cause_specific_mortality_rate"].data == 0.0
    assert (lookup_tables["life_expectancy"].data["value"] == 98.0).all()


def test_mortality_creates_columns(setup_sim_with_pop_and_mortality):
    sim, bp, mortality = setup_sim_with_pop_and_mortality
    pop = sim.get_population()
    expected_columns_created = set(mortality.columns_created)
    mortality_created_columns = set(pop.columns).difference(
        set(bp.columns_created + ["tracked"])
    )
    assert expected_columns_created == mortality_created_columns


def test_mortality_rate(setup_sim_with_pop_and_mortality):
    sim, bp, mortality = setup_sim_with_pop_and_mortality
    sim.step()
    pop1 = sim.get_population()
    mortality_rates = mortality.mortality_rate(pop1.index)["other_causes"]
    # Calculate mortality rate like component to cmpare
    lookup_tables = mortality.lookup_tables
    acmr = lookup_tables["all_cause_mortality_rate"](pop1.index)
    modeled_csmr = mortality.cause_specific_mortality_rate(pop1.index)
    unmodeled_csmr_raw = lookup_tables["unmodeled_cause_specific_mortality_rate"](pop1.index)
    unmodeled_csmr = mortality.unmodeled_csmr(pop1.index)
    expected_mortality_rates = (acmr - modeled_csmr - unmodeled_csmr_raw + unmodeled_csmr) * (
        sim._clock.step_size.days / 365
    )
    # Cannot compare two floats with FuzzyChecker
    assert np.isclose(mortality_rates, expected_mortality_rates).all()


def test_mortality_updates_population_columns(setup_sim_with_pop_and_mortality):
    sim, bp, mortality = setup_sim_with_pop_and_mortality
    pop0 = sim.get_population()
    sim.step()
    pop1 = sim.get_population()

    # Configrm mortalit7y component updated columns correctly
    # Note alive will be tested by finding the simulants that died
    columns_to_update = ["cause_of_death", "exit_time", "years_of_life_lost"]
    dead_idx = pop1.index[pop1["alive"] == "dead"]
    for col in columns_to_update:
        assert (pop1.loc[dead_idx, col] != pop0.loc[dead_idx, col]).all()
    assert (pop1.loc[dead_idx, "cause_of_death"] == "other_causes").all()
    assert pop1.loc[dead_idx, "exit_time"].unique()[0] == sim._clock._clock_time


def test_mortality_cause_of_death(
    fuzzy_checker, base_config, full_simulants, base_plugins, generate_population_mock
):

    start_population_size = len(full_simulants)
    generate_population_mock.return_value = full_simulants.drop(columns=["tracked"])
    bp = BasePopulation()
    mortality = Mortality()
    # Set up Disease model
    year_start = base_config.time.start.year
    year_end = base_config.time.end.year

    healthy = BaseDiseaseState("healthy")
    mort_get_data_funcs = {
        "dwell_time": lambda _, __: pd.Timedelta(days=0),
        "disability_weight": lambda _, __: 0.0,
        "prevalence": lambda _, __: build_table(
            0.5, year_start - 1, year_end, ["age", "year", "sex", "value"]
        ),
        "excess_mortality_rate": lambda _, __: build_table(0.7, year_start - 1, year_end),
    }
    mortality_state = DiseaseState("sick", get_data_functions=mort_get_data_funcs)
    healthy.add_transition(Transition(healthy, mortality_state))

    model = DiseaseModel("test", initial_state=healthy, states=[healthy, mortality_state])
    sim = InteractiveContext(
        components=[bp, mortality, model], plugin_configuration=base_plugins, setup=False
    )
    override_config = {
        "population": {
            "population_size": start_population_size,
            "include_sex": "Male",
        },
        "mortality": {
            "all_cause_mortality_rate": {
                "value": 0.8,
            }
        },
    }
    sim.configuration.update(override_config)
    sim.setup()
    sim.step()
    # Only other causes and sick for cause of death
    pop1 = sim.get_population()
    for cause_of_death in ["other_causes", "sick"]:
        dead = pop1.loc[pop1["cause_of_death"] == cause_of_death]
        # Disease model seems to set mortality rate for that diesease back to 0
        # if a simulant dies from it
        rates = mortality.mortality_rate(pop1.index)[cause_of_death].unique()
        for mortality_rate in rates:
            if mortality_rate == 0:
                continue
            else:
                rate = mortality_rate
        fuzzy_checker.fuzzy_assert_proportion(
            name=f"test_mortality_rate_{cause_of_death}",
            observed_numerator=len(dead),
            observed_denominator=len(pop1),
            target_proportion=rate,
        )


def test_mortality_ylls(setup_sim_with_pop_and_mortality):
    sim, bp, mortality = setup_sim_with_pop_and_mortality
    sim.step()
    pop1 = sim.get_population()

    dead_idx = pop1.index[pop1["alive"] == "dead"]
    ylls = pop1.loc[dead_idx, "years_of_life_lost"]
    assert (ylls == mortality.lookup_tables["life_expectancy"](dead_idx)).all()
