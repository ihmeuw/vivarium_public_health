"""
===================
Risk Exposure Model
===================

This module contains tools for modeling categorical and continuous risk
exposure.

"""
import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData

from vivarium_public_health.causal_factor.exposure import CausalFactor


class Risk(CausalFactor):
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
        super().__init__(risk)
        self.exposure_column_name = (
            f"{self.causal_factor.name}_exposure_for_non_loglinear_riskeffect"
        )

    VALID_ENTITY_TYPES = ["risk_factor", "alternative_risk_factor"]

    #################
    # Setup methods #
    #################

    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self.includes_non_loglinear_risk_effect = bool(
            [
                component
                for component in builder.components.list_components()
                if component.startswith(
                    f"non_log_linear_risk_effect.{self.causal_factor.name}_on_"
                )
            ]
        )
        if self.includes_non_loglinear_risk_effect:
            builder.population.register_initializer(
                initializer=self.initialize_exposure,
                columns=self.exposure_column_name,
                required_resources=[self.exposure_name],
            )

    def initialize_exposure(self, pop_data: SimulantData) -> None:
        exposure = self.get_exposure(pop_data.index)
        self.population_view.initialize(exposure)

    def on_time_step_prepare(self, event: Event) -> None:
        if self.includes_non_loglinear_risk_effect:
            exposure = self.get_exposure(event.index)
            self.population_view.update(
                self.exposure_column_name,
                lambda _: exposure,
            )

    def get_exposure(self, index: pd.Index) -> pd.Series:
        """Updates the exposure column with pipeline values.

        HACK: This is effectively caching the exposure pipeline for use by other
        components. Specifically, :meth:`vivarium_public_health.risks.effect.NonLogLinearRiskEffect.get_relative_risk_source`
        needs the exposure values but calling that pipeline was very slow. By
        maintaining a cached copy of the exposure values in a private column, we
        can then request that corresponding "simple" pipeline from the population
        view instead which is significantly faster.
        """
        exposure = self.population_view.get(index, self.exposure_name)
        exposure.name = self.exposure_column_name
        return exposure
