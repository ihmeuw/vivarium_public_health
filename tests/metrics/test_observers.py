from collections import namedtuple

import numpy as np
import pandas as pd
import pytest

from vivarium import InteractiveContext
from vivarium.framework.utilities import from_yearly
from vivarium.testing_utilities import TestPopulation, build_table, metadata

from vivarium_public_health.disease import BaseDiseaseState, DiseaseModel, DiseaseState, RiskAttributableDisease, RateTransition
from vivarium_public_health.metrics.disability import DisabilityObserver as DisabilityObserver_
from vivarium_public_health.metrics.stratification import ResultsStratifier as ResultsStratifier_

from vivarium_public_health.disease.state import SusceptibleState
from vivarium_public_health.disease.transition import TransitionString
from vivarium_public_health.population import Mortality


class ResultsStratifier(ResultsStratifier_):
    configuration_defaults = {
        "stratification": {
            "default": ["age_group", "sex"],
        }
    }


class DisabilityObserver(DisabilityObserver_):
    """Counts years lived with disability.

    By default, this counts both aggregate and cause-specific years lived
    with disability over the full course of the simulation.

    In the model specification, your configuration for this component should
    be specified as, e.g.:

    .. code-block:: yaml

        configuration:
            observers:
                disability:
                    exclude:
                        - "sex"
                    include:
                        - "sample_stratification"
    """

    configuration_defaults = {
        "stratification": {
            "disability": {
                "exclude": ["age_group"],
                "include": ["sex"],
            }
        }
    }

    # def setup(self, builder: Builder):
    #     super().setup()


def test_disability_observer_setup(mocker):
    """Test that DisabilityObserver.setup() registers expected observations
    and returns expected disease classes."""

    observer = DisabilityObserver()
    builder = mocker.Mock()
    builder.results.register_observation = mocker.Mock()

    # Set up fake calls for cause-specific register_observation args
    MockCause = namedtuple("MockCause", "state_id")
    builder.components.get_components_by_type = lambda n: [
        MockCause("flu"),
        MockCause("measles"),
    ]
    builder.value.get_value = lambda n: n

    builder.results.register_observation.assert_not_called()
    observer.setup(builder)
    builder.results.register_observation.assert_any_call(
        name="ylds_due_to_all_causes",
        pop_filter='tracked == True and alive == "alive"',
        aggregator_sources=[str(observer.disability_weight)],
        aggregator=observer._disability_weight_aggregator,
        requires_columns=["alive"],
        requires_values=["disability_weight"],
        additional_stratifications=observer.config.include,
        excluded_stratifications=observer.config.exclude,
        when="time_step__prepare",
    )
    builder.results.register_observation.assert_any_call(
        name="ylds_due_to_flu",
        pop_filter='tracked == True and alive == "alive"',
        aggregator_sources=[str("flu.disability_weight")],
        aggregator=observer._disability_weight_aggregator,
        requires_columns=["alive"],
        requires_values=["flu.disability_weight"],
        additional_stratifications=observer.config.include,
        excluded_stratifications=observer.config.exclude,
        when="time_step__prepare",
    )
    builder.results.register_observation.assert_any_call(
        name="ylds_due_to_measles",
        pop_filter='tracked == True and alive == "alive"',
        aggregator_sources=[str("measles.disability_weight")],
        aggregator=observer._disability_weight_aggregator,
        requires_columns=["alive"],
        requires_values=["measles.disability_weight"],
        additional_stratifications=observer.config.include,
        excluded_stratifications=observer.config.exclude,
        when="time_step__prepare",
    )
    assert builder.results.register_observation.call_count == 3
    assert DiseaseState in observer.disease_classes
    assert RiskAttributableDisease in observer.disease_classes


def test__disability_weight_aggregator():
    """Test that the disability weight aggregator produces expected ylds."""
    observer = DisabilityObserver()
    observer.step_size = pd.Timedelta(days=365.25)  # easy yld math
    fake_weights = pd.DataFrame(1.0, index=range(1000), columns=["disability_weight"])
    aggregated_weight = observer._disability_weight_aggregator(fake_weights)
    assert aggregated_weight == 1000.0


# @pytest.mark.parametrize("disability_weight_value", [0.0, 0.25, 0.5, 1.0,])
@pytest.mark.parametrize("disability_weight_value_0, disability_weight_value_1", [(0.25, 0.5)])
def test_disability_accumulation(base_config, base_plugins, disability_weight_value_0, disability_weight_value_1):
    """Integration test for the observer and the Results Management system."""
    year_start = base_config.time.start.year
    year_end = base_config.time.end.year

    time_step = pd.Timedelta(days=base_config.time.step_size)

    # Set up two disease models (_0 and _1), to test against multiple causes of disability
    healthy_0 = BaseDiseaseState("healthy_0")
    healthy_1 = BaseDiseaseState("healthy_1")  # TODO: Susceptible state change?
    disability_get_data_funcs_0 = {
        "disability_weight": lambda _, __: build_table(disability_weight_value_0, year_start - 1, year_end),
        "prevalence": lambda _, __: build_table(
            0.25, year_start - 1, year_end, ["age", "year", "sex", "value"]
        ),
    }
    disability_get_data_funcs_1 = {
        "disability_weight": lambda _, __: build_table(disability_weight_value_1, year_start - 1, year_end),
        "prevalence": lambda _, __: build_table(
            0.5, year_start - 1, year_end, ["age", "year", "sex", "value"]
        ),
    }
    disability_state_0 = DiseaseState("sick_cause_0", get_data_functions=disability_get_data_funcs_0)
    disability_state_1 = DiseaseState("sick_cause_1", get_data_functions=disability_get_data_funcs_1)
    model_0 = DiseaseModel("model_0", initial_state=healthy_0, states=[healthy_0, disability_state_0])
    model_1 = DiseaseModel("model_1", initial_state=healthy_1, states=[healthy_1, disability_state_1])

    simulation = InteractiveContext(
        components=[TestPopulation(), model_0, model_1, ResultsStratifier(), DisabilityObserver()],
        configuration=base_config,
        plugin_configuration=base_plugins,
    )
    simulation.step()

    pop = simulation.get_population()
    sub_pop = {"healthy": pop[(pop["model_0"] == "healthy_0") & (pop["model_1"] == "healthy_1")],
               "sick_0":    pop[(pop["model_0"] == "sick_cause_0") & (pop["model_1"] == "healthy_1")],
               "sick_1":    pop[(pop["model_0"] == "healthy_0") & (pop["model_1"] == "sick_cause_1")],
               "sick_0_1":  pop[(pop["model_0"] == "sick_cause_0") & (pop["model_1"] == "sick_cause_1")]}

    # Get pipelines
    disability_weight = simulation._values.get_value("disability_weight")
    disability_weight_0 = simulation._values.get_value("sick_cause_0.disability_weight")
    disability_weight_1 = simulation._values.get_value("sick_cause_1.disability_weight")

    time_step / pd.Timedelta("365.25 days")
    for sub_pop_key in ["healthy", "sick_0", "sick_1"]:
        assert np.isclose(disability_weight(sub_pop[sub_pop_key].index),(disability_weight_0(sub_pop[sub_pop_key].index) * time_step/pd.Timedelta("365.25 days")
                + disability_weight_1(sub_pop[sub_pop_key].index) * time_step/pd.Timedelta("365.25 days")), rtol=0.001).all()

    # TODO: Add check for disability weight calc with two weights

    # TODO: check the metrics pipeline
    #  Check that all keys and values are expected
