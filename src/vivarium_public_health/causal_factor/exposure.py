"""
==============
Exposure Model
==============

This module contains tools for modeling categorical and continuous exposures.

"""
from abc import ABC
from typing import Any

import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.population import SimulantData
from vivarium.framework.randomness import RandomnessStream

from vivarium_public_health.causal_factor.distributions import (
    CausalFactorDistribution,
    ContinuousDistribution,
    DichotomousDistribution,
    EnsembleDistribution,
    PolytomousDistribution,
)
from vivarium_public_health.causal_factor.utilities import get_exposure_post_processor
from vivarium_public_health.utilities import EntityString


class CausalFactor(Component, ABC):
    """A model for an attribute defined by either a continuous or a categorical value.

    For example,

    #. high systolic blood pressure as an attribute where the SBP is not dichotomized
       into hypotension and normal but is treated as the actual SBP
       measurement.
    #. smoking as two categories: current smoker and non-smoker.

    This component can source data either from builder.data or from parameters
    supplied in the configuration. If data is derived from the configuration, it
    must be an integer or float expressing the desired exposure level or a
    covariate name that is intended to be used as a proxy. For example, for a
    causal factor named "causal_factor", the configuration could look like this:

    .. code-block:: yaml

       configuration:
           causal_factor:
               exposure: 1.0

    or

    .. code-block:: yaml

       configuration:
           causal_factor:
               exposure: proxy_covariate

    For polytomous causal factors, you can also provide an optional 'rebinned_exposed'
    block in the configuration to indicate that the causal factor should be rebinned
    into a dichotomous causal factor. That block should contain a list of the categories
    that should be rebinned into a single exposed category in the resulting
    dichotomous causal factor. For example, for a causal factor named "causal_factor" with categories
    cat1, cat2, cat3, and cat4 that you wished to rebin into a dichotomous causal factor
    with an exposed category containing cat1 and cat2 and an unexposed category
    containing cat3 and cat4, the configuration could look like this:

    .. code-block:: yaml

       configuration:
           causal_factor:
              rebinned_exposed: ['cat1', 'cat2']

    For alternative risk factors, you must provide a 'category_thresholds'
    block in the in configuration to dictate the thresholds that should be
    used to bin the continuous distributions. Note that this is mutually
    exclusive with providing 'rebinned_exposed' categories. For a causal factor named
    "causal_factor", the configuration could look like:

    .. code-block:: yaml

       configuration:
           causal_factor:
               category_thresholds: [7, 8, 9]

    """

    exposure_distributions: dict[str, CausalFactorDistribution] = {
        "dichotomous": DichotomousDistribution,
        "ordered_polytomous": PolytomousDistribution,
        "unordered_polytomous": PolytomousDistribution,
        "normal": ContinuousDistribution,
        "lognormal": ContinuousDistribution,
        "ensemble": EnsembleDistribution,
    }

    VALID_ENTITY_TYPES = []

    ##############
    # Properties #
    ##############

    @property
    def name(self) -> str:
        """The name of this causal factor component."""
        return self.causal_factor

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        """Default configuration values for this causal factor component.

        Configuration structure::

            {causal_factor_name}:
                data_sources:
                    exposure:
                        Source for exposure data. Default is the artifact key
                        ``{causal_factor}.exposure``.
                    ensemble_distribution_weights:
                        Source for ensemble distribution weights (only used
                        for ensemble distributions). Default is the artifact
                        key ``{causal_factor}.exposure_distribution_weights``.
                    exposure_standard_deviation:
                        Source for exposure standard deviation data (only used
                        for continuous distributions). Default is the artifact
                        key ``{causal_factor}.exposure_standard_deviation``.
                distribution_type: str
                    Type of exposure distribution. Can be one of:
                    ``"dichotomous"``, ``"ordered_polytomous"``,
                    ``"unordered_polytomous"``, ``"normal"``, ``"lognormal"``,
                    or ``"ensemble"``. Default loads from artifact at
                    ``{causal_factor}.distribution``.
                rebinned_exposed: list[str]
                    Categories to combine into a single "exposed" category
                    when rebinning a polytomous causal factor to dichotomous. Only
                    used with polytomous distributions. Default is empty
                    list (no rebinning).
                category_thresholds: list[float]
                    Thresholds for converting continuous distributions to
                    categorical. Mutually exclusive with ``rebinned_exposed``.
                    Default is empty list (no categorization).
        """
        return {
            self.name: {
                "data_sources": {
                    "exposure": f"{self.causal_factor}.exposure",
                    "ensemble_distribution_weights": f"{self.causal_factor}.exposure_distribution_weights",
                    "exposure_standard_deviation": f"{self.causal_factor}.exposure_standard_deviation",
                },
                "distribution_type": f"{self.causal_factor}.distribution",
                # rebinned_exposed only used for DichotomousDistribution
                "rebinned_exposed": [],
                "category_thresholds": [],
            }
        }

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, causal_factor: str):
        """

        Parameters
        ----------
        causal_factor
            The type and name of a causal factor, specified as "type.name". Type is singular.
        """
        super().__init__()
        self.causal_factor = EntityString(causal_factor)
        self._validate_entity_type()

        self.distribution_type = None

        self.randomness_stream_name = f"initial_{self.causal_factor.name}_propensity"
        self.propensity_name = f"{self.causal_factor.name}.propensity"
        self.exposure_name = f"{self.causal_factor.name}.exposure"
        self.exposure_column_name = (
            f"{self.causal_factor.name}_exposure_for_non_loglinear_effect"
        )

    def _validate_entity_type(self) -> None:
        """Validates that the entity type of the causal factor is supported."""
        if self.causal_factor.type not in self.VALID_ENTITY_TYPES:
            raise ValueError(
                f"Entity type must be one of {self.VALID_ENTITY_TYPES}, "
                f"but got '{self.causal_factor.type}' for '{self.causal_factor}'."
            )

    #################
    # Setup methods #
    #################

    def setup(self, builder: Builder) -> None:
        """Set up the causal factor component.

        Determine the distribution type, create the exposure distribution,
        obtain a randomness stream, register the exposure pipeline, and
        register a propensity initializer.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.
        """
        self._components = builder.components
        self.distribution_type = self.get_distribution_type(builder)
        self.exposure_distribution = self.get_exposure_distribution(builder)

        self.randomness = self.get_randomness_stream(builder)
        self.register_exposure_pipeline(builder)

        builder.population.register_initializer(
            initializer=self.initialize_propensity,
            columns=self.propensity_name,
            required_resources=[self.randomness],
        )

    def get_distribution_type(self, builder: Builder) -> str:
        """Get the distribution type for the causal factor from the configuration.

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

    def get_exposure_distribution(self, builder: Builder) -> CausalFactorDistribution:
        """Create and set up the exposure distribution component for the causal
        factor based on its distribution type.

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
            distribution_class = self.exposure_distributions[self.distribution_type]
            distribution: CausalFactorDistribution = distribution_class(
                self.causal_factor, self.distribution_type
            )
        except KeyError:
            raise NotImplementedError(
                f"Distribution type {self.distribution_type} is not supported."
            )
        distribution.setup_component(builder)
        return distribution

    def get_randomness_stream(self, builder: Builder) -> RandomnessStream:
        """Return a randomness stream for propensity initialization.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.

        Returns
        -------
            The randomness stream for this causal factor.
        """
        return builder.randomness.get_stream(self.randomness_stream_name)

    def register_exposure_pipeline(self, builder: Builder) -> None:
        """Register the exposure pipeline with the simulation.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.
        """
        builder.value.register_attribute_producer(
            self.exposure_name,
            source=[self.exposure_distribution.exposure_ppf_pipeline],
            preferred_post_processor=get_exposure_post_processor(builder, self.name),
        )

    ########################
    # Event-driven methods #
    ########################

    def initialize_propensity(self, pop_data: SimulantData) -> None:
        """Initialize propensity values for new simulants.

        Parameters
        ----------
        pop_data
            Metadata about the simulants being initialized.
        """
        propensity = pd.Series(
            self.randomness.get_draw(pop_data.index), name=self.propensity_name
        )
        self.population_view.initialize(propensity)
