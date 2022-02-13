"""
===================
Risk Exposure Model
===================

This module contains tools for modeling categorical and continuous risk
exposure.

"""
from typing import Dict, List

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.population import PopulationView, SimulantData
from vivarium.framework.randomness import RandomnessStream
from vivarium.framework.values import Pipeline

from vivarium_public_health.risks.data_transformations import (
    get_exposure_post_processor,
)
from vivarium_public_health.risks.distributions import SimulationDistribution
from vivarium_public_health.utilities import EntityString


class Risk:
    """A model for a risk factor defined by either a continuous or a categorical
    value. For example,

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

    configuration_defaults = {
        "risk": {
            "exposure": "data",
            "rebinned_exposed": [],
            "category_thresholds": [],
        }
    }

    def __init__(self, risk: str):
        """
        Parameters
        ----------
        risk :
            the type and name of a risk, specified as "type.name". Type is singular.
        """
        self.risk = EntityString(risk)
        self.configuration_defaults = self._get_configuration_defaults()
        self.exposure_distribution = self._get_exposure_distribution()
        self._sub_components = [self.exposure_distribution]

        self._randomness_stream_name = f"initial_{self.risk.name}_propensity"
        self.propensity_column_name = f"{self.risk.name}_propensity"
        self.propensity_pipeline_name = f"{self.risk.name}.propensity"
        self.exposure_pipeline_name = f"{self.risk.name}.exposure"

    def __repr__(self) -> str:
        return f"Risk({self.risk})"

    ##########################
    # Initialization methods #
    ##########################

    def _get_configuration_defaults(self) -> Dict[str, Dict]:
        return {self.risk.name: Risk.configuration_defaults["risk"]}

    def _get_exposure_distribution(self) -> SimulationDistribution:
        return SimulationDistribution(self.risk)

    ##############
    # Properties #
    ##############

    @property
    def name(self) -> str:
        return f"risk.{self.risk}"

    @property
    def sub_components(self) -> List:
        return self._sub_components

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.randomness = self._get_randomness_stream(builder)
        self.propensity = self._get_propensity_pipeline(builder)
        self.exposure = self._get_exposure_pipeline(builder)
        self.population_view = self._get_population_view(builder)

        self._register_simulant_initializer(builder)

    def _get_randomness_stream(self, builder) -> RandomnessStream:
        return builder.randomness.get_stream(self._randomness_stream_name)

    def _get_propensity_pipeline(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.propensity_pipeline_name,
            source=lambda index: (
                self.population_view.subview([self.propensity_column_name])
                .get(index)
                .squeeze(axis=1)
            ),
            requires_columns=[self.propensity_column_name],
        )

    def _get_exposure_pipeline(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.exposure_pipeline_name,
            source=self._get_current_exposure,
            requires_columns=["age", "sex"],
            requires_values=[self.propensity_pipeline_name],
            preferred_post_processor=get_exposure_post_processor(builder, self.risk),
        )

    def _get_population_view(self, builder: Builder) -> PopulationView:
        return builder.population.get_view([self.propensity_column_name])

    def _register_simulant_initializer(self, builder: Builder) -> None:
        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            creates_columns=[self.propensity_column_name],
            requires_streams=[self._randomness_stream_name],
        )

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        self.population_view.update(
            pd.Series(
                self.randomness.get_draw(pop_data.index), name=self.propensity_column_name
            )
        )

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def _get_current_exposure(self, index: pd.Index) -> pd.Series:
        propensity = self.propensity(index)
        return pd.Series(self.exposure_distribution.ppf(propensity), index=index)
