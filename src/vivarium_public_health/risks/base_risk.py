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
from vivarium.framework.population import SimulantData
from vivarium.framework.randomness import RandomnessStream

from vivarium_public_health.risks.data_transformations import get_exposure_post_processor
from vivarium_public_health.risks.distributions import (
    ContinuousDistribution,
    DichotomousDistribution,
    EnsembleDistribution,
    PolytomousDistribution,
    RiskExposureDistribution,
)
from vivarium_public_health.utilities import EntityString


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
        """Provides default configuration values for this risk component.

        Configuration structure::

            {risk_name}:
                data_sources:
                    exposure:
                        Source for exposure data. Default is the artifact key
                        ``{risk}.exposure``.
                    ensemble_distribution_weights:
                        Source for ensemble distribution weights (only used
                        for ensemble distributions). Default is the artifact
                        key ``{risk}.exposure_distribution_weights``.
                    exposure_standard_deviation:
                        Source for exposure standard deviation data (only used
                        for continuous distributions). Default is the artifact
                        key ``{risk}.exposure_standard_deviation``.
                distribution_type: str
                    Type of exposure distribution. Can be one of:
                    ``"dichotomous"``, ``"ordered_polytomous"``,
                    ``"unordered_polytomous"``, ``"normal"``, ``"lognormal"``,
                    or ``"ensemble"``. Default loads from artifact at
                    ``{risk}.distribution``.
                rebinned_exposed: list[str]
                    Categories to combine into a single "exposed" category
                    when rebinning a polytomous risk to dichotomous. Only
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
        self.propensity_name = f"{self.risk.name}.propensity"
        self.exposure_name = f"{self.risk.name}.exposure"

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self._components = builder.components
        self.distribution_type = self.get_distribution_type(builder)
        self.exposure_distribution = self.get_exposure_distribution(builder)

        self.randomness = self.get_randomness_stream(builder)
        self.register_exposure_pipeline(builder)
        builder.population.register_initializer(
            initializer=self.on_initialize_simulants,
            columns=self.propensity_name,
            required_resources=[self.randomness],
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
        # HACK / FIXME [MIC-6756]: Because we need to start setting up each Risk to know
        # its corresponding RiskExposureDistribution type, we cannot rely on sub-components.
        # Instead, we've determined the RiskExposureDistribution here and want to set it
        # up manually which requires temporarily changing the current component
        # in the component manager.
        self._components._manager._current_component = exposure_distribution
        exposure_distribution.setup_component(builder)
        self._components._manager._current_component = self
        return exposure_distribution

    def get_randomness_stream(self, builder: Builder) -> RandomnessStream:
        return builder.randomness.get_stream(self.randomness_stream_name)

    def register_exposure_pipeline(self, builder: Builder) -> None:
        builder.value.register_attribute_producer(
            self.exposure_name,
            source=[self.exposure_distribution.exposure_ppf_pipeline],
            preferred_post_processor=get_exposure_post_processor(builder, self.name),
        )

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        propensity = pd.Series(
            self.randomness.get_draw(pop_data.index), name=self.propensity_name
        )
        self.population_view.update(propensity)
