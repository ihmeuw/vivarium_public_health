"""
===================
Disability Observer
===================

This module contains tools for observing years lived with disability (YLDs)
in the simulation.

"""

from typing import List

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.results import StratifiedObserver
from vivarium.framework.values import Pipeline, list_combiner, union_post_processor

from vivarium_public_health.disease import DiseaseState, RiskAttributableDisease
from vivarium_public_health.metrics.reporters import COLUMNS, write_dataframe_to_csv
from vivarium_public_health.utilities import to_years


class DisabilityObserver(StratifiedObserver):
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

    ##############
    # Properties #
    ##############

    @property
    def disease_classes(self) -> List:
        return [DiseaseState, RiskAttributableDisease]

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self):
        super().__init__()
        self.disability_weight_pipeline_name = "disability_weight"

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.config = builder.configuration.stratification.disability
        self.step_size = pd.Timedelta(days=builder.configuration.time.step_size)
        self.disability_weight = self.get_disability_weight_pipeline(builder)

    #################
    # Setup methods #
    #################

    def register_observations(self, builder: Builder) -> None:
        cause_states = builder.components.get_components_by_type(tuple(self.disease_classes))

        builder.results.register_observation(
            name="ylds_due_to_all_causes",
            pop_filter='tracked == True and alive == "alive"',
            aggregator_sources=[self.disability_weight_pipeline_name],
            aggregator=self.disability_weight_aggregator,
            requires_columns=["alive"],
            requires_values=["disability_weight"],
            additional_stratifications=self.config.include,
            excluded_stratifications=self.config.exclude,
            when="time_step__prepare",
            report=self.report,
        )

        for cause_state in cause_states:
            cause_disability_weight_pipeline_name = (
                f"{cause_state.state_id}.disability_weight"
            )
            builder.results.register_observation(
                name=f"ylds_due_to_{cause_state.state_id}",
                pop_filter='tracked == True and alive == "alive"',
                aggregator_sources=[cause_disability_weight_pipeline_name],
                aggregator=self.disability_weight_aggregator,
                requires_columns=["alive"],
                requires_values=[cause_disability_weight_pipeline_name],
                additional_stratifications=self.config.include,
                excluded_stratifications=self.config.exclude,
                when="time_step__prepare",
                report=self.report,
            )

    def get_disability_weight_pipeline(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.disability_weight_pipeline_name,
            source=lambda index: [pd.Series(0.0, index=index)],
            preferred_combiner=list_combiner,
            preferred_post_processor=union_post_processor,
        )

    ###############
    # Aggregators #
    ###############

    def disability_weight_aggregator(self, dw: pd.DataFrame) -> float:
        return (dw * to_years(self.step_size)).sum().squeeze()

    ##################
    # Report methods #
    ##################

    def report(
        self,
        measure: str,
        results: pd.DataFrame,
    ) -> None:
        """Combine the measure-specific observer results and save to a single file."""
        measure, cause = [s.strip("_") for s in measure.split("due_to")]
        write_dataframe_to_csv(
            measure,
            results,
            self.results_dir,
            self.random_seed,
            self.input_draw,
            {COLUMNS.CAUSE: cause},
        )
