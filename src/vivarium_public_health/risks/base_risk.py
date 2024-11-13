"""
===================
Risk Exposure Model
===================

This module contains tools for modeling categorical and continuous risk
exposure.

"""

from typing import Any

import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.randomness import RandomnessStream
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


class Risk(Component):
    """A model for a risk factor defined by either a continuous or a categorical value.

    For example,

    #. high systolic blood pressure as a risk where the SBP is not dichotomized
       into hypotension and normal but is treated as the actual SBP
       measurement.
    #. smoking as two categories: current smoker and non-smoker.

    This component can source data either from builder.data or from parameters
    supplied in the configuration. If data is derived from the configuration, it
    must be an integer or float expressing the desired exposure level or a
    covariate name that is intended to be used as a proxy. For example, for a
    risk named "risk", the configuration could look like this:

    .. code-block:: yaml

       configuration:
           risk:
               exposure: 1.0

    or

    .. code-block:: yaml

       configuration:
           risk:
               exposure: proxy_covariate

    For polytomous risks, you can also provide an optional 'rebinned_exposed'
    block in the configuration to indicate that the risk should be rebinned
    into a dichotomous risk. That block should contain a list of the categories
    that should be rebinned into a single exposed category in the resulting
    dichotomous risk. For example, for a risk named "risk" with categories
    cat1, cat2, cat3, and cat4 that you wished to rebin into a dichotomous risk
    with an exposed category containing cat1 and cat2 and an unexposed category
    containing cat3 and cat4, the configuration could look like this:

    .. code-block:: yaml

       configuration:
           risk:
              rebinned_exposed: ['cat1', 'cat2']

    For alternative risk factors, you must provide a 'category_thresholds'
    block in the in configuration to dictate the thresholds that should be
    used to bin the continuous distributions. Note that this is mutually
    exclusive with providing 'rebinned_exposed' categories. For a risk named
    "risk", the configuration could look like:

    .. code-block:: yaml

       configuration:
           risk:
               category_thresholds: [7, 8, 9]

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
        return self.risk

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        return {
            self.name: {
                "data_sources": {
                    "exposure": f"{self.risk}.exposure",
                    "ensemble_distribution_weights": f"{self.risk}.exposure_distribution_weights",
                    "exposure_standard_deviation": f"{self.risk}.exposure_standard_deviation",
                },
                "distribution_type": f"{self.risk}.distribution",
                # rebinned_exposed only used for DichotomousDistribution
                "rebinned_exposed": [],
                "category_thresholds": [],
            }
        }

    @property
    def columns_created(self) -> list[str]:
        columns_to_create = [self.propensity_column_name]
        if self.create_exposure_column:
            columns_to_create.append(self.exposure_column_name)
        return columns_to_create

    @property
    def initialization_requirements(self) -> dict[str, list[str]]:
        return {
            "requires_columns": [],
            "requires_values": [],
            "requires_streams": [self.randomness_stream_name],
        }

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, risk: str):
        """

        Parameters
        ----------
        risk
            the type and name of a risk, specified as "type.name". Type is singular.
        """
        super().__init__()
        self.risk = EntityString(risk)
        self.distribution_type = None

        self.randomness_stream_name = f"initial_{self.risk.name}_propensity"
        self.propensity_column_name = f"{self.risk.name}_propensity"
        self.propensity_pipeline_name = f"{self.risk.name}.propensity"
        self.exposure_pipeline_name = f"{self.risk.name}.exposure"
        self.exposure_column_name = f"{self.risk.name}_exposure"

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
        self.exposure = self.get_exposure_pipeline(builder)

        # We want to set this to True iff there is a non-loglinear risk effect
        # on this risk instance
        self.create_exposure_column = bool(
            [
                component
                for component in builder.components.list_components()
                if component.startswith(f"non_log_linear_risk_effect.{self.risk.name}_on_")
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
                self.risk, self.distribution_type
            )
        except KeyError:
            raise NotImplementedError(
                f"Distribution type {self.distribution_type} is not supported."
            )

        exposure_distribution.setup_component(builder)
        return exposure_distribution

    def get_randomness_stream(self, builder: Builder) -> RandomnessStream:
        return builder.randomness.get_stream(self.randomness_stream_name)

    def get_propensity_pipeline(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.propensity_pipeline_name,
            source=lambda index: (
                self.population_view.subview([self.propensity_column_name])
                .get(index)
                .squeeze(axis=1)
            ),
            requires_columns=[self.propensity_column_name],
        )

    def get_exposure_pipeline(self, builder: Builder) -> Pipeline:
        required_columns = get_lookup_columns(
            self.exposure_distribution.lookup_tables.values()
        )
        return builder.value.register_value_producer(
            self.exposure_pipeline_name,
            source=self.get_current_exposure,
            requires_columns=required_columns,
            requires_values=[
                self.propensity_pipeline_name,
                self.exposure_distribution.parameters_pipeline_name,
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
        self.update_exposure_column(pop_data.index)

    def on_time_step_prepare(self, event: Event) -> None:
        self.update_exposure_column(event.index)

    def update_exposure_column(self, index: pd.Index) -> None:
        if self.create_exposure_column:
            exposure = pd.Series(self.exposure(index), name=self.exposure_column_name)
            self.population_view.update(exposure)

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def get_current_exposure(self, index: pd.Index) -> pd.Series:
        propensity = self.propensity(index)
        return pd.Series(self.exposure_distribution.ppf(propensity), index=index)
