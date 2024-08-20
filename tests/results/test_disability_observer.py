import itertools
import re

import numpy as np
import pandas as pd
import pytest
from layered_config_tree import LayeredConfigTree
from vivarium import InteractiveContext
from vivarium.testing_utilities import TestPopulation

from tests.test_utilities import build_table_with_age
from vivarium_public_health.disease import DiseaseModel, DiseaseState, RiskAttributableDisease
from vivarium_public_health.disease.state import SusceptibleState
from vivarium_public_health.results.columns import COLUMNS
from vivarium_public_health.results.disability import (
    DisabilityObserver as DisabilityObserver_,
)
from vivarium_public_health.results.stratification import ResultsStratifier


# Subclass of DisabilityObserver for integration testing
class DisabilityObserver(DisabilityObserver_):
    @property
    def configuration_defaults(self):
        return {
            "stratification": {
                "disability": {
                    "exclude": ["age_group"],
                    "include": ["sex"],
                }
            }
        }


def test_disability_observer_setup(mocker):
    """Test that DisabilityObserver.setup() registers expected observations
    and returns expected disease classes."""

    observer = DisabilityObserver_()
    builder = mocker.Mock()
    mocker.patch("vivarium.component.Component.build_all_lookup_tables")
    mocker.patch("vivarium.component.Component.get_configuration")
    builder.results.register_adding_observation = mocker.Mock()
    builder.configuration.time.step_size = 28
    builder.configuration.output_data.results_directory = "some/results/directory"
    builder.configuration.stratification.excluded_categories = LayeredConfigTree(
        {"disability": []}
    )

    # Set up fake calls for cause-specific register_observation args
    flu = DiseaseState("flu")
    measles = DiseaseState("measles")
    builder.components.get_components_by_type = lambda n: [flu, measles]
    builder.value.get_value = lambda n: n

    builder.results.register_adding_observation.assert_not_called()
    observer.setup_component(builder)

    assert builder.results.register_adding_observation.call_count == 1
    cause_pipelines = [
        "flu.disability_weight",
        "measles.disability_weight",
        "all_causes.disability_weight",
    ]
    builder.results.register_adding_observation.assert_any_call(
        name="ylds",
        pop_filter='tracked == True and alive == "alive"',
        when="time_step__prepare",
        requires_columns=["alive"],
        requires_values=cause_pipelines,
        results_formatter=observer.format_results,
        additional_stratifications=observer.configuration.include,
        excluded_stratifications=observer.configuration.exclude,
        aggregator_sources=cause_pipelines,
        aggregator=observer.disability_weight_aggregator,
    )

    assert set(observer.disability_classes) == set([DiseaseState, RiskAttributableDisease])


def test__disability_weight_aggregator():
    """Test that the disability weight aggregator produces expected ylds."""
    observer = DisabilityObserver_()
    observer.step_size = pd.Timedelta(days=365.25)  # easy yld math
    fake_weights = pd.DataFrame(1.0, index=range(1000), columns=["disability_weight"])
    aggregated_weight = observer.disability_weight_aggregator(fake_weights)
    assert aggregated_weight == 1000.0


@pytest.mark.parametrize(
    "disability_weight_value_0, disability_weight_value_1",
    [(0.25, 0.5), (0.99, 0.1), (0.1, 0.1)],
)
def test_disability_accumulation(
    base_config,
    base_plugins,
    disability_weight_value_0,
    disability_weight_value_1,
):
    """Integration test for the disability observer and the Results Management system."""
    year_start = base_config.time.start.year
    year_end = base_config.time.end.year
    time_step = pd.Timedelta(days=base_config.time.step_size)

    # Set up two disease models (_0 and _1), to test against multiple causes of disability
    healthy_0 = SusceptibleState("healthy_0")
    healthy_1 = SusceptibleState("healthy_1")
    disability_get_data_funcs_0 = {
        "disability_weight": lambda _, __: build_table_with_age(
            disability_weight_value_0,
            parameter_columns={"year": (year_start - 1, year_end)},
        ),
        "prevalence": lambda _, __: build_table_with_age(
            0.45, parameter_columns={"year": (year_start - 1, year_end)}
        ),
    }
    disability_get_data_funcs_1 = {
        "disability_weight": lambda _, __: build_table_with_age(
            disability_weight_value_1,
            parameter_columns={"year": (year_start - 1, year_end)},
        ),
        "prevalence": lambda _, __: build_table_with_age(
            0.65, parameter_columns={"year": (year_start - 1, year_end)}
        ),
    }
    disability_state_0 = DiseaseState(
        "sick_cause_0", get_data_functions=disability_get_data_funcs_0
    )
    disability_state_1 = DiseaseState(
        "sick_cause_1", get_data_functions=disability_get_data_funcs_1
    )
    # TODO: Add test against using a RiskAttributableDisease in addition to a DiseaseModel
    model_0 = DiseaseModel(
        "model_0", initial_state=healthy_0, states=[healthy_0, disability_state_0]
    )
    model_1 = DiseaseModel(
        "model_1", initial_state=healthy_1, states=[healthy_1, disability_state_1]
    )

    # Add the results dir since we didn't go through cli.py
    simulation = InteractiveContext(
        components=[
            TestPopulation(),
            model_0,
            model_1,
            ResultsStratifier(),
            DisabilityObserver(),
        ],
        configuration=base_config,
        plugin_configuration=base_plugins,
    )

    # Take two time steps (not just one in order to ensure metrics are updating properly)
    simulation.step()
    simulation.step()

    pop = simulation.get_population()
    sub_pop_mask = {
        "healthy": (pop["model_0"] == "healthy_0") & (pop["model_1"] == "healthy_1"),
        "sick_0": (pop["model_0"] == "sick_cause_0") & (pop["model_1"] == "healthy_1"),
        "sick_1": (pop["model_0"] == "healthy_0") & (pop["model_1"] == "sick_cause_1"),
        "sick_0_1": (pop["model_0"] == "sick_cause_0") & (pop["model_1"] == "sick_cause_1"),
    }

    # Get pipelines
    disability_weight_all_causes = simulation.get_value("all_causes.disability_weight")
    disability_weight_0 = simulation.get_value("sick_cause_0.disability_weight")
    disability_weight_1 = simulation.get_value("sick_cause_1.disability_weight")

    # Check that disability weights are computed as expected
    for sub_pop_key in ["healthy", "sick_0", "sick_1", "sick_0_1"]:
        assert np.isclose(
            disability_weight_all_causes(pop[sub_pop_mask[sub_pop_key]].index),
            (
                1
                - (
                    (1 - disability_weight_0(pop[sub_pop_mask[sub_pop_key]].index))
                    * (1 - disability_weight_1(pop[sub_pop_mask[sub_pop_key]].index))
                )
            ),
            rtol=0.0000001,
        ).all()

    # Test that metrics are correct
    results = simulation.get_results()["ylds"]

    # yld_masks format: {cause: (state, filter, dw_pipeline)}
    yld_masks = {
        "all_causes": (None, slice(None), disability_weight_all_causes),
        "model_0": ("sick_cause_0", pop["model_0"] == "sick_cause_0", disability_weight_0),
        "model_1": ("sick_cause_1", pop["model_1"] == "sick_cause_1", disability_weight_1),
    }

    # Check that all expected observations are there
    assert set(zip(results["sex"], results[COLUMNS.ENTITY])) == set(
        itertools.product(*[["Female", "Male"], list(yld_masks)])
    )
    for cause, (state, *_) in yld_masks.items():
        if state:
            assert (
                results.loc[results[COLUMNS.ENTITY] == cause, COLUMNS.SUB_ENTITY] == state
            ).all()
        else:  # all_causes
            assert (
                results.loc[results[COLUMNS.ENTITY] == cause, COLUMNS.SUB_ENTITY]
                == "all_causes"
            ).all()

    assert set(results.columns) == set(
        [
            "sex",
            COLUMNS.MEASURE,
            COLUMNS.ENTITY_TYPE,
            COLUMNS.ENTITY,
            COLUMNS.SUB_ENTITY,
            COLUMNS.VALUE,
        ]
    )
    assert (results[COLUMNS.MEASURE] == "ylds").all()
    assert (results[COLUMNS.ENTITY_TYPE] == "cause").all()

    # Check that all the yld values are as expected
    time_scale = time_step / pd.Timedelta("365.25 days") * 2
    for cause, (state, pop_filter, dw) in yld_masks.items():
        cause_specific_pop = pop[pop_filter]
        for sex in ["Female", "Male"]:
            sub_pop = cause_specific_pop[cause_specific_pop["sex"] == sex]
            expected_ylds = (dw(sub_pop.index) * time_scale).sum()
            actual_ylds = results.loc[
                (results[COLUMNS.ENTITY] == cause) & (results["sex"] == sex), COLUMNS.VALUE
            ].values
            assert len(actual_ylds) == 1
            assert np.isclose(expected_ylds, actual_ylds[0], rtol=0.0000001)


@pytest.mark.parametrize("exclusions", [[], ["flu"], ["all_causes"], ["flu", "all_causes"]])
def test_set_causes_of_disability(exclusions, mocker):
    observer = DisabilityObserver_()

    builder = mocker.Mock()
    mocker.patch("vivarium.component.Component.build_all_lookup_tables")
    builder.configuration.time.step_size = 28
    builder.configuration.stratification.excluded_categories = LayeredConfigTree(
        {"disability": exclusions}
    )

    # Set up fake calls for cause-specific register_observation args
    flu = DiseaseState("flu")
    builder.components.get_components_by_type = lambda n: [flu]

    observer.setup_component(builder)
    assert {cause.state_id for cause in observer.causes_of_disability} == {
        "flu",
        "all_causes",
    } - set(exclusions)


def test_set_causes_of_disability_raises(mocker):
    observer = DisabilityObserver_()

    builder = mocker.Mock()
    mocker.patch("vivarium.component.Component.build_all_lookup_tables")
    builder.configuration.time.step_size = 28
    builder.configuration.stratification.excluded_categories = LayeredConfigTree(
        {"disability": ["arthritis"]}  # not an instantiated disease
    )

    # Set up fake calls for cause-specific register_observation args (but NOT 'arthritis')
    flu = DiseaseState("flu")
    builder.components.get_components_by_type = lambda n: [flu]

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Excluded 'disability' causes {'arthritis'} not found in expected categories categories: ['flu', 'all_causes']"
        ),
    ):
        observer.setup_component(builder)


@pytest.mark.parametrize(
    "exclusions",
    [
        [],
        ["all_causes"],
        ["all_causes", "sick_cause_0"],
    ],
)
def test_category_exclusions(
    base_config,
    base_plugins,
    exclusions,
):
    """Integration test for the disability observer and the Results Management system."""
    year_start = base_config.time.start.year
    year_end = base_config.time.end.year

    # Set up two disease models (_0 and _1), to test against multiple causes of disability
    healthy_0 = SusceptibleState("healthy_0")
    healthy_1 = SusceptibleState("healthy_1")
    disability_get_data_funcs_0 = {
        "disability_weight": lambda _, __: build_table_with_age(
            0.99,
            parameter_columns={"year": (year_start - 1, year_end)},
        ),
        "prevalence": lambda _, __: build_table_with_age(
            0.45, parameter_columns={"year": (year_start - 1, year_end)}
        ),
    }
    disability_get_data_funcs_1 = {
        "disability_weight": lambda _, __: build_table_with_age(
            0.1,
            parameter_columns={"year": (year_start - 1, year_end)},
        ),
        "prevalence": lambda _, __: build_table_with_age(
            0.65, parameter_columns={"year": (year_start - 1, year_end)}
        ),
    }
    disability_state_0 = DiseaseState(
        "sick_cause_0", get_data_functions=disability_get_data_funcs_0
    )
    disability_state_1 = DiseaseState(
        "sick_cause_1", get_data_functions=disability_get_data_funcs_1
    )
    model_0 = DiseaseModel(
        "model_0", initial_state=healthy_0, states=[healthy_0, disability_state_0]
    )
    model_1 = DiseaseModel(
        "model_1", initial_state=healthy_1, states=[healthy_1, disability_state_1]
    )

    # Add exclusions to model spec
    base_config.update(
        {
            "stratification": {
                "excluded_categories": {
                    "disability": exclusions,
                }
            }
        }
    )
    simulation = InteractiveContext(
        components=[
            TestPopulation(),
            model_0,
            model_1,
            ResultsStratifier(),
            DisabilityObserver(),
        ],
        configuration=base_config,
        plugin_configuration=base_plugins,
    )

    simulation.step()
    results = simulation.get_results()["ylds"]
    assert set(results["sub_entity"]) == {"sick_cause_0", "sick_cause_1", "all_causes"} - set(
        exclusions
    )
