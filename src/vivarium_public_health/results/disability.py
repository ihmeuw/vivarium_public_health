"""
====================
Disability Observers
====================

This module contains tools for observing years lived with disability (YLDs)
in the simulation.

"""

import pandas as pd
from layered_config_tree import LayeredConfigTree
from loguru import logger
from pandas.api.types import CategoricalDtype
from vivarium.framework.engine import Builder
from vivarium.framework.values import Pipeline, list_combiner, union_post_processor

from vivarium_public_health.disease import DiseaseState, RiskAttributableDisease
from vivarium_public_health.results.columns import COLUMNS
from vivarium_public_health.results.observer import PublicHealthObserver
from vivarium_public_health.results.simple_cause import SimpleCause
from vivarium_public_health.utilities import to_years


class DisabilityObserver(PublicHealthObserver):
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
    Attributes
    ----------
    disability_weight_pipeline_name
        The name of the pipeline that produces disability weights.
    step_size
        The time step size of the simulation.
    disability_weight
        The pipeline that produces disability weights.
    causes_of_disability
        The causes of disability to be observed.

    """

    ##############
    # Properties #
    ##############

    @property
    def disability_classes(self) -> list[type]:
        """The classes to be considered as causes of disability."""
        return [DiseaseState, RiskAttributableDisease]

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self) -> None:
        super().__init__()
        self.disability_weight_pipeline_name = "all_causes.disability_weight"

    #################
    # Setup methods #
    #################

    def setup(self, builder: Builder) -> None:
        """Set up the observer."""
        self.step_size = pd.Timedelta(days=builder.configuration.time.step_size)
        self.disability_weight = self.get_disability_weight_pipeline(builder)
        self.set_causes_of_disability(builder)

    def set_causes_of_disability(self, builder: Builder) -> None:
        """Set the causes of disability to be observed.

        The causes to be observed are any registered components of class types
        found in the ``disability_classes`` property *excluding* any listed in
        the model spec as ``excluded_categories``.

        Notes
        -----
        We implement exclusions here instead of during the stratification call
        like most other categories because disabilities are unique in that they are
        *not* actually registered stratifications.

        Also note that we add an 'all_causes' category here.
        """
        causes_of_disability = builder.components.get_components_by_type(
            self.disability_classes
        )
        # Convert to SimpleCause instances and add on all_causes
        causes_of_disability = [
            SimpleCause.create_from_specific_cause(cause) for cause in causes_of_disability
        ] + [SimpleCause("all_causes", "all_causes", "cause")]

        excluded_causes = (
            builder.configuration.stratification.excluded_categories.to_dict().get(
                "disability", []
            )
        )

        # Handle exclusions that don't exist in the list of causes
        cause_names = [cause.state_id for cause in causes_of_disability]
        unknown_exclusions = set(excluded_causes) - set(cause_names)
        if len(unknown_exclusions) > 0:
            raise ValueError(
                f"Excluded 'disability' causes {unknown_exclusions} not found in "
                f"expected categories categories: {cause_names}"
            )

        # Drop excluded causes
        if excluded_causes:
            logger.debug(
                f"'disability' has category exclusion requests: {excluded_causes}\n"
                "Removing these from the allowable categories."
            )
        self.causes_of_disability = [
            cause for cause in causes_of_disability if cause.state_id not in excluded_causes
        ]

    def get_configuration(self, builder: Builder) -> LayeredConfigTree:
        """Get the stratification configuration for this observer.

        Parameters
        ----------
        builder
            The builder object for the simulation.

        Returns
        -------
            The stratification configuration for this observer.
        """
        return builder.configuration.stratification.disability

    def register_observations(self, builder: Builder) -> None:
        """Register an observation for years lived with disability."""
        cause_pipelines = [
            f"{cause.state_id}.disability_weight" for cause in self.causes_of_disability
        ]
        self.register_adding_observation(
            builder=builder,
            name="ylds",
            pop_filter='tracked == True and alive == "alive"',
            when="time_step__prepare",
            requires_columns=["alive"],
            requires_values=cause_pipelines,
            additional_stratifications=self.configuration.include,
            excluded_stratifications=self.configuration.exclude,
            aggregator_sources=cause_pipelines,
            aggregator=self.disability_weight_aggregator,
        )

    def get_disability_weight_pipeline(self, builder: Builder) -> Pipeline:
        """Register (and return) the pipeline that produces disability weights.

        Parameters
        ----------
        builder
            The builder object for the simulation.

        Returns
        -------
            The pipeline that produces disability weights.
        """
        return builder.value.register_value_producer(
            self.disability_weight_pipeline_name,
            source=lambda index: [pd.Series(0.0, index=index)],
            preferred_combiner=list_combiner,
            preferred_post_processor=union_post_processor,
        )

    ###############
    # Aggregators #
    ###############

    def disability_weight_aggregator(self, dw: pd.DataFrame) -> float | pd.Series:
        """Aggregate disability weights for the time step.

        Parameters
        ----------
        dw
            The disability weights to aggregate.

        Returns
        -------
            The aggregated disability weights.
        """
        aggregated_dw = (dw * to_years(self.step_size)).sum().squeeze()
        if isinstance(aggregated_dw, pd.Series):
            aggregated_dw.index.name = "cause_of_disability"
        return aggregated_dw

    ##############################
    # Results formatting methods #
    ##############################

    def format(self, measure: str, results: pd.DataFrame) -> pd.DataFrame:
        """Format wide YLD results to match typical/long stratified results.

        YLDs are unique in that we can't stratify by cause of disability (because
        there can be multiple at once), and so the results here are actually wide
        by disability weight pipeline name. This method formats the results to be
        long by cause of disability.

        Parameters
        ----------
        measure
            The measure.
        results
            The wide results to format.

        Returns
        -------
            The results stacked by causes of disability.

        Notes
        -----
        This method also adds the 'sub_entity' column to the results.
        """
        if len(self.causes_of_disability) > 1:
            # Drop the unused 'value' column and rename the remaining pipeline names to cause names
            results = results.rename(
                columns={
                    col: col.replace(".disability_weight", "") for col in results.columns
                }
            )[[col.state_id for col in self.causes_of_disability]]
        else:
            # Rename the 'value' column to the single cause of disability
            results = results.rename(
                columns={COLUMNS.VALUE: self.causes_of_disability[0].state_id}
            )
        # Get desired index names prior to stacking
        idx_names = list(results.index.names) + [COLUMNS.SUB_ENTITY]
        results = pd.DataFrame(results.stack(), columns=[COLUMNS.VALUE])
        # Name the new index level
        results.index.set_names(idx_names, inplace=True)
        results = results.reset_index()
        results[COLUMNS.SUB_ENTITY] = results[COLUMNS.SUB_ENTITY].astype(CategoricalDtype())
        return results

    def get_entity_type_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        """Get the 'entity_type' column values."""
        entity_type_map = {
            cause.state_id: cause.cause_type for cause in self.causes_of_disability
        }
        return results[COLUMNS.SUB_ENTITY].map(entity_type_map).astype(CategoricalDtype())

    def get_entity_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        """Get the 'entity' column values."""
        entity_map = {cause.state_id: cause.model for cause in self.causes_of_disability}
        return results[COLUMNS.SUB_ENTITY].map(entity_map).astype(CategoricalDtype())

    def get_sub_entity_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        """Get the 'sub_entity' column values."""
        # The sub-entity col was created in the 'format' method
        return results[COLUMNS.SUB_ENTITY]
