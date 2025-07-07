from abc import ABC, abstractmethod

import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.randomness import RandomnessStream
from vivarium.framework.resource import Resource
from vivarium.framework.values import Pipeline

from vivarium_public_health.risks.data_transformations import get_exposure_post_processor
from vivarium_public_health.risks.distributions import (
    ContinuousDistribution,
    DichotomousDistribution,
    EnsembleDistribution,
    PolytomousDistribution,
    RiskExposureDistribution,
)
from vivarium_public_health.utilities import EntityString, get_lookup_columns

__ALLOWABLE_LEVEL_TYPES = ["exposure", "coverage"]


class HealthFactor(Component, ABC):
    """A base class to store common functionality for for different health factors.

    This class is used to define the determinant of models health factors such as
    risks and the exposure to these risks, or interventions and the available coverage
    for these interventions.

    """

    exposure_distributions = {
        "dichotomous": DichotomousDistribution,
        "ordered_polytomous": PolytomousDistribution,
        "unordered_polytomous": PolytomousDistribution,
        "normal": ContinuousDistribution,
        "lognormal": ContinuousDistribution,
        "ensemble": EnsembleDistribution,
    }

    ##############
    # Properties #
    ##############

    @property
    def name(self) -> str:
        return self.health_factor

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        return {
            self.name: {
                "data_sources": {
                    f"{self.level_type}": f"{self.health_factor}.{self.level_type}",
                    "ensemble_distribution_weights": f"{self.health_factor}.exposure_distribution_weights",
                    "exposure_standard_deviation": f"{self.health_factor}.exposure_standard_deviation",
                },
                "distribution_type": f"{self.health_factor}.distribution",
                # rebinned_exposed only used for DichotomousDistribution
                "rebinned_exposed": [],
                "category_thresholds": [],
            }
        }

    @property
    def columns_created(self) -> list[str]:
        columns_to_create = [self.propensity_column_name]
        if self.create_exposure_column:
            columns_to_create.append(self.determinant_column_name)
        return columns_to_create

    @property
    def initialization_requirements(self) -> list[str | Resource]:
        return [self.randomness]

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, health_factor: str, level_type: str = "exposure") -> None:
        """

        Parameters
        ----------
        health_factor
            the type and name of a health factor, specified as "type.name". Type is singular.
        level_type
            The type of level for the health factor, e.g., "exposure" or "coverage".
        """
        super().__init__()
        self.health_factor = EntityString(health_factor)
        self.distribution_type = None
        self.level_type = level_type
        if self.level_type not in __ALLOWABLE_LEVEL_TYPES:
            raise ValueError(
                f"Invalid level type '{self.level_type}' for {self.name}. "
                f"Allowed types are: {__ALLOWABLE_LEVEL_TYPES}."
            )

        self.randomness_stream_name = f"initial_{self.health_factor.name}_propensity"
        self.propensity_column_name = f"{self.health_factor.name}_propensity"
        self.propensity_pipeline_name = f"{self.health_factor.name}.propensity"
        self.determinant_pipeline_name = f"{self.health_factor.name}.{self.level_type}"
        self.determinant_column_name = f"{self.health_factor.name}_{self.level_type}"

    #################
    # Setup methods #
    #################

    def build_all_lookup_tables(self, builder: "Builder") -> None:
        # All lookup tables are built in the exposure distribution
        pass

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.distribution_type = self.get_distribution_type(builder)
        self.exposure_distribution = self.get_exposure_distribution(builder)

        self.randomness = self.get_randomness_stream(builder)
        self.propensity = self.get_propensity_pipeline(builder)
        self.determinant = self.get_determinant_pipeline(builder)

        # We want to set this to True iff there is a non-loglinear risk effect
        # on this risk instance
        self.create_exposure_column = bool(
            [
                component
                for component in builder.components.list_components()
                if component.startswith(
                    f"non_log_linear_risk_effect.{self.health_factor.name}_on_"
                )
            ]
        )

    def get_distribution_type(self, builder: Builder) -> str:
        """Get the distribution type for the risk from the configuration.

        If the configured distribution type is not one of the supported types,
        it is assumed to be a data source and the data is retrieved using the
        get_data method.

        Parameters
        ----------
        builder
            The builder object.

        Returns
        -------
            The distribution type.
        """
        if self.configuration is None:
            self.configuration = self.get_configuration(builder)

        distribution_type = self.configuration["distribution_type"]
        if distribution_type not in self.exposure_distributions.keys():
            # todo deal with incorrect typing
            distribution_type = self.get_data(builder, distribution_type)

        if self.configuration["rebinned_exposed"]:
            if distribution_type != "dichotomous" or "polytomous" not in distribution_type:
                raise ValueError(
                    f"Unsupported risk distribution type '{distribution_type}' "
                    f"for {self.name}. Rebinned exposed categories are only "
                    "supported for dichotomous and polytomous distributions."
                )
            distribution_type = "dichotomous"
        return distribution_type

    def get_exposure_distribution(self, builder: Builder) -> RiskExposureDistribution:
        """Creates and sets up the exposure distribution component for the Risk
        based on its distribution type.

        Parameters
        ----------
        builder
            The builder object.

        Returns
        -------
            The exposure distribution.

        Raises
        ------
        NotImplementedError
            If the distribution type is not supported.
        """
        try:
            exposure_distribution = self.exposure_distributions[self.distribution_type](
                self.health_factor, self.distribution_type
            )
        except KeyError:
            raise NotImplementedError(
                f"Distribution type {self.distribution_type} is not supported."
            )

        exposure_distribution.setup_component(builder)
        return exposure_distribution

    def get_randomness_stream(self, builder: Builder) -> RandomnessStream:
        return builder.randomness.get_stream(self.randomness_stream_name, component=self)

    def get_propensity_pipeline(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.propensity_pipeline_name,
            source=lambda index: (
                self.population_view.subview([self.propensity_column_name])
                .get(index)
                .squeeze(axis=1)
            ),
            component=self,
            required_resources=[self.propensity_column_name],
        )

    def get_determinant_pipeline(self, builder: Builder) -> Pipeline:
        required_columns = get_lookup_columns(
            self.exposure_distribution.lookup_tables.values()
        )
        return builder.value.register_value_producer(
            self.determinant_pipeline_name,
            source=self.get_current_determinant,
            component=self,
            required_resources=required_columns
            + [
                self.propensity,
                self.exposure_distribution.exposure_parameters,
            ],
            preferred_post_processor=get_exposure_post_processor(builder, self.name),
        )

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        propensity = pd.Series(
            self.randomness.get_draw(pop_data.index), name=self.propensity_column_name
        )
        self.population_view.update(propensity)
        self.update_determinant_column(pop_data.index)

    def on_time_step_prepare(self, event: Event) -> None:
        self.update_determinant_column(event.index)

    def update_determinant_column(self, index: pd.Index) -> None:
        if self.create_exposure_column:
            exposure = pd.Series(self.determinant(index), name=self.determinant_column_name)
            self.population_view.update(exposure)

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def get_current_determinant(self, index: pd.Index) -> pd.Series:
        propensity = self.propensity(index)
        return pd.Series(self.exposure_distribution.ppf(propensity), index=index)
