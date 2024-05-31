"""
===================
Disability Observer
===================

This module contains tools for observing years lived with disability (YLDs)
in the simulation.

"""

from typing import Any, List, Union

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.results import StratifiedObserver
from vivarium.framework.values import Pipeline, list_combiner, union_post_processor

from vivarium_public_health.disease import DiseaseState, RiskAttributableDisease
from vivarium_public_health.metrics.columns import COLUMNS
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
    def disease_classes(self) -> List[Any]:
        return [DiseaseState, RiskAttributableDisease]

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self) -> None:
        super().__init__()
        self.disability_weight_pipeline_name = "disability_weight"

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.config = builder.configuration.stratification.disability
        self.step_size = pd.Timedelta(days=builder.configuration.time.step_size)
        self.disability_weight = self.get_disability_weight_pipeline(builder)
        self.causes_of_disease = [
            cause
            for cause in builder.components.get_components_by_type(
                tuple(self.disease_classes)
            )
        ]

    #################
    # Setup methods #
    #################

    def register_observations(self, builder: Builder) -> None:
        cause_pipelines = [self.disability_weight_pipeline_name] + [
            f"{cause.state_id}.disability_weight" for cause in self.causes_of_disease
        ]
        builder.results.register_observation(
            name="ylds",
            pop_filter='tracked == True and alive == "alive"',
            aggregator_sources=cause_pipelines,
            aggregator=self.disability_weight_aggregator,
            requires_columns=["alive"],
            requires_values=cause_pipelines,
            additional_stratifications=self.config.include,
            excluded_stratifications=self.config.exclude,
            when="time_step__prepare",
            format_results=self.format_results,
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

    def disability_weight_aggregator(self, dw: pd.DataFrame) -> Union[float, pd.Series]:
        aggregated_dw = (dw * to_years(self.step_size)).sum().squeeze()
        if isinstance(aggregated_dw, pd.Series):
            aggregated_dw.index.name = "cause_of_disability"
        return aggregated_dw

    ##################
    # Report methods #
    ##################

    def format_results(
        self,
        measure: str,
        results: pd.DataFrame,
    ) -> pd.DataFrame:
        """Format results. Note that ylds are unique in that we
        can't stratify by cause of disability (because there can be multiple at
        once), and so the results here are actually wide by disability weight
        pipeline name.
        """

        # Drop the unused 'value' column and rename the pipeline names to causes
        results = (
            results.drop(columns=["value"])
            .rename(columns={"disability_weight": "all_causes"})
            .rename(
                columns={
                    col: col.replace(".disability_weight", "") for col in results.columns
                },
            )
        )

        # Stack the causes of disability
        idx_names = list(results.index.names) + [COLUMNS.SUB_ENTITY]
        results = pd.DataFrame(results.stack(), columns=[COLUMNS.VALUE])
        # Name the new index level
        results.index.set_names(idx_names, inplace=True)
        results = results.reset_index()

        results[COLUMNS.MEASURE] = measure
        results[COLUMNS.ENTITY_TYPE] = "cause"
        results.loc[
            results[COLUMNS.SUB_ENTITY] == "all_causes", COLUMNS.ENTITY
        ] = "all_causes"
        for cause in self.causes_of_disease:
            cause_mask = results[COLUMNS.SUB_ENTITY] == cause.state_id
            results.loc[cause_mask, COLUMNS.ENTITY] = cause.model
            results.loc[cause_mask, COLUMNS.ENTITY_TYPE] = cause.cause_type
        results["random_seed"] = self.random_seed
        results["input_draw"] = self.input_draw

        return results[[c for c in results.columns if c != COLUMNS.VALUE] + [COLUMNS.VALUE]]
