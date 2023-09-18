"""
===================
Risk Exposure Model
===================

This module contains tools for modeling categorical and continuous risk
exposure.

"""
from typing import Any, Dict, List

import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.population import SimulantData
from vivarium.framework.randomness import RandomnessStream
from vivarium.framework.values import Pipeline

from vivarium_public_health.risks.data_transformations import (
    get_exposure_post_processor,
)
from vivarium_public_health.risks.distributions import SimulationDistribution
from vivarium_public_health.utilities import EntityString


class Risk(Component):
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

    CONFIGURATION_DEFAULTS = {
        "risk": {
            "exposure": "data",
            "rebinned_exposed": [],
            "category_thresholds": [],
        }
    }

    ##############
    # Properties #
    ##############

    @property
    def configuration_defaults(self) -> Dict[str, Any]:
        return {self.risk.name: self.CONFIGURATION_DEFAULTS["risk"]}

    @property
    def columns_created(self) -> List[str]:
        return [self.propensity_column_name]

    @property
    def initialization_requirements(self) -> Dict[str, List[str]]:
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
        risk :
            the type and name of a risk, specified as "type.name". Type is singular.
        """
        super().__init__()
        self.risk = EntityString(risk)
        self.exposure_distribution = self.get_exposure_distribution()
        self._sub_components = [self.exposure_distribution]

        self.randomness_stream_name = f"initial_{self.risk.name}_propensity"
        self.propensity_column_name = f"{self.risk.name}_propensity"
        self.propensity_pipeline_name = f"{self.risk.name}.propensity"
        self.exposure_pipeline_name = f"{self.risk.name}.exposure"

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.randomness = self.get_randomness_stream(builder)
        self.propensity = self.get_propensity_pipeline(builder)
        self.exposure = self.get_exposure_pipeline(builder)

    ##########################
    # Initialization methods #
    ##########################

    def get_exposure_distribution(self) -> SimulationDistribution:
        return SimulationDistribution(self.risk)

    #################
    # Setup methods #
    #################

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
        return builder.value.register_value_producer(
            self.exposure_pipeline_name,
            source=self.get_current_exposure,
            requires_columns=["age", "sex"],
            requires_values=[self.propensity_pipeline_name],
            preferred_post_processor=get_exposure_post_processor(builder, self.risk),
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

    def get_current_exposure(self, index: pd.Index) -> pd.Series:
        propensity = self.propensity(index)
        return pd.Series(self.exposure_distribution.ppf(propensity), index=index)
