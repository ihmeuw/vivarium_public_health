import numpy as np
import pandas as pd
import pytest
from vivarium import Component, InteractiveContext
from vivarium.framework.lookup.table import ScalarTable
from vivarium.framework.state_machine import Transition
from vivarium_testing_utils import FuzzyChecker

from tests.test_utilities import build_table_with_age
from vivarium_public_health.disease import BaseDiseaseState, DiseaseModel, DiseaseState
from vivarium_public_health.population import BasePopulation


@pytest.fixture
def setup_sim_with_pop_and_mortality(
    full_simulants, base_plugins, generate_population_mock
) -> tuple[InteractiveContext, Component, Component]:

    # Initializes an Interactive context with BasePopulation and Mortality components
    start_population_size = len(full_simulants)

    generate_population_mock.return_value = full_simulants
    bp = BasePopulation()
    sim = InteractiveContext(components=[bp], plugin_configuration=base_plugins, setup=False)
    override_config = {
        "population": {
            "population_size": start_population_size,
            "include_sex": "Male",
        },
        "mortality": {"unmodeled_causes": []},
    }
    sim.configuration.update(override_config)
    sim.setup()
    return sim, bp, sim.list_components()["mortality"]


def test_mortality_default_lookup_configuration(setup_sim_with_pop_and_mortality):
    _, __, mortality = setup_sim_with_pop_and_mortality

    assert (mortality.acmr_table.data["value"] == 0.5).all()
    assert mortality.unmodeled_csmr_table.data == 0.0
    assert (mortality.life_expectancy_table.data["value"] == 98.0).all()


def test_mortality_creates_attributes(setup_sim_with_pop_and_mortality):
    sim, bp, mortality = setup_sim_with_pop_and_mortality
    pop = sim.get_population()
    expected_columns_created = mortality.private_columns
    expected_attributes_created = [
        mortality.mortality_rate_pipeline,
        mortality.cause_specific_mortality_rate_pipeline,
        mortality.unmodeled_csmr_pipeline,
        mortality.unmodeled_csmr_paf_pipeline,
    ]
    # the time manager, BasePopulation, AgedOutSimulants, and Disability create attributes themselves
    other_columns_created = bp.private_columns + [
        "is_aged_out",
        "simulant_step_size",
        "all_causes.disability_weight",
    ]
    mortality_created_columns = [
        col for col in pop.columns.get_level_values(0) if col not in other_columns_created
    ]
    assert set(expected_columns_created + expected_attributes_created) == set(
        mortality_created_columns
    )


def test_mortality_rate(setup_sim_with_pop_and_mortality):
    sim, _, mortality = setup_sim_with_pop_and_mortality
    sim.step()
    mortality_rates = sim.get_population("mortality_rate")
    # Calculate mortality rate like component to cmpare
    pop_idx = mortality_rates.index
    acmr = mortality.acmr_table(pop_idx)
    modeled_csmr = sim.get_population("cause_specific_mortality_rate")
    unmodeled_csmr_raw = mortality.unmodeled_csmr_table(pop_idx)
    unmodeled_csmr = sim.get_population("affected_unmodeled.cause_specific_mortality_rate")
    expected_mortality_rates = (acmr - modeled_csmr - unmodeled_csmr_raw + unmodeled_csmr) * (
        sim._clock.step_size.days / 365
    )
    # Cannot compare two floats with FuzzyChecker
    assert np.isclose(mortality_rates, expected_mortality_rates).all()


def test_mortality_updates_population_columns(setup_sim_with_pop_and_mortality):
    sim, bp, mortality = setup_sim_with_pop_and_mortality
    update_columns = ["cause_of_death", "exit_time", "years_of_life_lost"]
    pop0 = sim.get_population(update_columns + ["is_alive"])
    sim.step()
    pop1 = sim.get_population(update_columns + ["is_alive"])

    # Check mortality component updates columns correctly
    # Note alive will be tested by finding the simulants that died
    dead_idx = pop1.index[pop1["is_alive"] == False]
    for col in update_columns:
        assert (pop1.loc[dead_idx, col] != pop0.loc[dead_idx, col]).all()
    assert (pop1.loc[dead_idx, "cause_of_death"] == "other_causes").all()
    # Only 1 time step taken
    assert len(pop1.loc[dead_idx, "exit_time"].unique()) == 1
    assert pop1.loc[dead_idx, "exit_time"].unique()[0] == sim._clock._clock_time


def test_mortality_cause_of_death(
    fuzzy_checker: FuzzyChecker,
    base_config,
    full_simulants,
    base_plugins,
    generate_population_mock,
):

    start_population_size = len(full_simulants)
    generate_population_mock.return_value = full_simulants
    bp = BasePopulation()
    # Set up Disease model
    year_start = base_config.time.start.year
    year_end = base_config.time.end.year

    healthy = BaseDiseaseState("healthy")
    mort_get_data_funcs = {
        "dwell_time": lambda _, __: pd.Timedelta(days=0),
        "disability_weight": lambda _, __: 0.0,
        "prevalence": lambda _, __: build_table_with_age(
            0.5, parameter_columns={"year": (year_start - 1, year_end)}
        ),
        "excess_mortality_rate": lambda _, __: build_table_with_age(
            0.7, parameter_columns={"year": (year_start - 1, year_end)}
        ),
    }
    mortality_state = DiseaseState("sick", get_data_functions=mort_get_data_funcs)
    healthy.add_transition(Transition(healthy, mortality_state))

    model = DiseaseModel("test", residual_state=healthy, states=[healthy, mortality_state])
    sim = InteractiveContext(
        components=[bp, model], plugin_configuration=base_plugins, setup=False
    )
    override_config = {
        "population": {
            "population_size": start_population_size,
            "include_sex": "Male",
        },
        "mortality": {"data_sources": {"all_cause_mortality_rate": 0.8}},
    }
    sim.configuration.update(override_config)
    sim.setup()
    mortality_rates = sim.get_population("mortality_rate")
    sim.step()
    # Only 'other_causes' and 'sick' for cause of death
    cause_of_death = sim.get_population("cause_of_death")
    for cause in ["other_causes", "sick"]:
        dead = cause_of_death.loc[cause_of_death == cause]
        # Disease model seems to set mortality rate for that disease back to 0
        # if a simulant dies from it
        rates = mortality_rates[cause].unique()
        for mortality_rate in rates:
            if mortality_rate == 0:
                continue
            else:
                mortality_rate = mortality_rate
            if cause == "sick":
                mortality_rate *= 0.5  # prevalence
            fuzzy_checker.fuzzy_assert_proportion(
                name=f"test_mortality_rate_{cause}",
                observed_numerator=len(dead),
                observed_denominator=len(cause_of_death),
                target_proportion=mortality_rate,
            )


def test_mortality_ylls(setup_sim_with_pop_and_mortality):
    sim, bp, mortality = setup_sim_with_pop_and_mortality
    sim.step()
    pop1 = sim.get_population(["is_alive", "years_of_life_lost"])

    dead_idx = pop1.index[pop1["is_alive"] == False]
    ylls = pop1.loc[dead_idx, "years_of_life_lost"]
    assert (ylls == mortality.life_expectancy_table(dead_idx)).all()
    alive_idx = pop1.index[pop1["is_alive"] == True]
    no_ylls = pop1.loc[alive_idx, "years_of_life_lost"]
    assert (no_ylls == 0).all()


def test_no_unmodeled_causes(setup_sim_with_pop_and_mortality):
    _, __, mortality = setup_sim_with_pop_and_mortality
    # No unmodeled causes by default
    assert isinstance(mortality.unmodeled_csmr_table, ScalarTable)
    assert mortality.unmodeled_csmr_table.data == 0.0


def test_unmodeled_causes(full_simulants, base_plugins, generate_population_mock):
    start_population_size = len(full_simulants)
    generate_population_mock.return_value = full_simulants
    bp = BasePopulation()

    sim = InteractiveContext(components=[bp], plugin_configuration=base_plugins, setup=False)
    override_config = {
        "population": {
            "population_size": start_population_size,
            "include_sex": "Male",
        },
        "mortality": {
            "unmodeled_causes": ["low_birth_weight", "malnutrition", "malaria"],
        },
    }
    sim.configuration.update(override_config)
    sim.setup()
    sim.step()
    other_causes_mortality_rate = sim.get_population("mortality_rate")
    assert len(other_causes_mortality_rate.unique()) == 1
    assert other_causes_mortality_rate.unique()[0] * 365 == 0.5
