import itertools
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from vivarium import InteractiveContext
from vivarium.framework.results import VALUE_COLUMN
from vivarium.testing_utilities import TestPopulation

from tests.test_utilities import build_table_with_age
from vivarium_public_health.disease import (
    DiseaseModel,
    DiseaseState,
    RiskAttributableDisease,
)
from vivarium_public_health.disease.state import SusceptibleState
from vivarium_public_health.metrics.disability import (
    DisabilityObserver as DisabilityObserver_,
)
from vivarium_public_health.metrics.stratification import ResultsStratifier


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
    builder.results.register_observation = mocker.Mock()
    builder.configuration.time.step_size = 28
    builder.configuration.output_data.results_directory = "some/results/directory"

    # Set up fake calls for cause-specific register_observation args
    MockCause = namedtuple("MockCause", "state_id")
    builder.components.get_components_by_type = lambda n: [
        MockCause("flu"),
        MockCause("measles"),
    ]
    builder.value.get_value = lambda n: n

    builder.results.register_observation.assert_not_called()
    observer.setup_component(builder)
    # Need to mock the results_dir since we didn't run this from cli
    builder.results.register_observation.assert_any_call(
        name="ylds_due_to_all_causes",
        pop_filter='tracked == True and alive == "alive"',
        aggregator_sources=["disability_weight"],
        aggregator=observer.disability_weight_aggregator,
        requires_columns=["alive"],
        requires_values=["disability_weight"],
        additional_stratifications=observer.config.include,
        excluded_stratifications=observer.config.exclude,
        when="time_step__prepare",
        report=observer.report,
    )
    builder.results.register_observation.assert_any_call(
        name="ylds_due_to_flu",
        pop_filter='tracked == True and alive == "alive"',
        aggregator_sources=["flu.disability_weight"],
        aggregator=observer.disability_weight_aggregator,
        requires_columns=["alive"],
        requires_values=["flu.disability_weight"],
        additional_stratifications=observer.config.include,
        excluded_stratifications=observer.config.exclude,
        when="time_step__prepare",
        report=observer.report,
    )
    builder.results.register_observation.assert_any_call(
        name="ylds_due_to_measles",
        pop_filter='tracked == True and alive == "alive"',
        aggregator_sources=["measles.disability_weight"],
        aggregator=observer.disability_weight_aggregator,
        requires_columns=["alive"],
        requires_values=["measles.disability_weight"],
        additional_stratifications=observer.config.include,
        excluded_stratifications=observer.config.exclude,
        when="time_step__prepare",
        report=observer.report,
    )
    assert builder.results.register_observation.call_count == 3
    assert DiseaseState in observer.disease_classes
    assert RiskAttributableDisease in observer.disease_classes


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
    tmpdir,
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
    model_0 = DiseaseModel(
        "model_0", initial_state=healthy_0, states=[healthy_0, disability_state_0]
    )
    model_1 = DiseaseModel(
        "model_1", initial_state=healthy_1, states=[healthy_1, disability_state_1]
    )

    # Add the results dir since we didn't go through cli.py
    results_dir = Path(tmpdir)
    base_config.update({"output_data": {"results_directory": str(results_dir)}})
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

    pop = simulation.get_population()
    sub_pop_mask = {
        "healthy": (pop["model_0"] == "healthy_0") & (pop["model_1"] == "healthy_1"),
        "sick_0": (pop["model_0"] == "sick_cause_0") & (pop["model_1"] == "healthy_1"),
        "sick_1": (pop["model_0"] == "healthy_0") & (pop["model_1"] == "sick_cause_1"),
        "sick_0_1": (pop["model_0"] == "sick_cause_0") & (pop["model_1"] == "sick_cause_1"),
    }

    # Get pipelines
    disability_weight = simulation.get_value("disability_weight")
    disability_weight_0 = simulation.get_value("sick_cause_0.disability_weight")
    disability_weight_1 = simulation.get_value("sick_cause_1.disability_weight")

    # Check that disability weights are computed as expected
    for sub_pop_key in ["healthy", "sick_0", "sick_1", "sick_0_1"]:
        assert np.isclose(
            disability_weight(pop[sub_pop_mask[sub_pop_key]].index),
            (
                1
                - (
                    (1 - disability_weight_0(pop[sub_pop_mask[sub_pop_key]].index))
                    * (1 - disability_weight_1(pop[sub_pop_mask[sub_pop_key]].index))
                )
            ),
            rtol=0.0000001,
        ).all()

    # Test that metrics are saved out correctly
    simulation.finalize()
    simulation.report()
    results_files = list(results_dir.rglob("*.csv"))
    assert set(file.name for file in results_files) == set(["ylds.csv"])
    results = pd.read_csv(results_files[0])

    # yld_masks format: {cause: (filter, dw_pipeline)}
    yld_masks = {
        "all_causes": (slice(None), disability_weight),
        "sick_cause_0": (pop["model_0"] == "sick_cause_0", disability_weight_0),
        "sick_cause_1": (pop["model_1"] == "sick_cause_1", disability_weight_1),
    }

    # Check that all expected observations are there
    assert set(zip(results["sex"], results["cause"])) == set(
        itertools.product(*[["Female", "Male"], list(yld_masks)])
    )

    # Check other columns (NOTE: no input_draw defined so shouldn't be there)
    assert set(results.columns) == set(
        ["sex", "cause", "measure", "random_seed", VALUE_COLUMN]
    )
    assert (results["measure"] == "ylds").all()
    assert (results["random_seed"] == 0).all()

    # Check that all the yld values are as expected
    time_scale = time_step / pd.Timedelta("365.25 days")
    for cause in ["all_causes", "sick_cause_0", "sick_cause_1"]:
        pop_filter, dw = yld_masks[cause]
        cause_specific_pop = pop[pop_filter]
        for sex in ["Female", "Male"]:
            sub_pop = cause_specific_pop[cause_specific_pop["sex"] == sex]
            expected_ylds = (dw(sub_pop.index) * time_scale).sum()
            actual_ylds = results.loc[
                (results["cause"] == cause) & (results["sex"] == sex), VALUE_COLUMN
            ].values
            assert len(actual_ylds) == 1
            assert np.isclose(expected_ylds, actual_ylds[0], rtol=0.0000001)
