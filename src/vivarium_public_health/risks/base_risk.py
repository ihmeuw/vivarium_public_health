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
    """Risk factor model defined by either a continuous or a categorical exposure value.

    For example:

    1. Model high systolic blood pressure as a risk where the SBP is not dichotomized
       into hypotension and normal but is treated as the actual SBP measurement.
    2. Model smoking as two categories: current smoker and non-smoker.

    This component can source data either from builder.data or from parameters
    supplied in the configuration. If data is derived from the configuration,
    provide an integer or float expressing the desired exposure level or a
    covariate name to use as a proxy. For example, for a risk named
    "some_risk", the configuration could look like this:

    .. code-block:: yaml

       configuration:
           some_risk:
               exposure: 1.0

    or

    .. code-block:: yaml

       configuration:
           some_risk:
               exposure: proxy_covariate

    For polytomous risks, optionally provide a 'rebinned_exposed' block in the
    configuration to indicate that the risk should be rebinned into a
    dichotomous risk. That block should contain a list of the categories to
    rebin into a single exposed category in the resulting dichotomous risk.
    For example, for a risk named "some_risk" with categories cat1, cat2,
    cat3, and cat4 that you wish to rebin into a dichotomous risk with an
    exposed category containing cat1 and cat2 and an unexposed category
    containing cat3 and cat4, the configuration could look like this:

    .. code-block:: yaml

       configuration:
           some_risk:
              rebinned_exposed: ['cat1', 'cat2']

    For alternative risk factors, provide a 'category_thresholds' block in
    the configuration to dictate the thresholds to use to bin the continuous
    distributions. Note that this is mutually exclusive with providing
    'rebinned_exposed' categories. For a risk named "some_risk", the
    configuration could look like:

    .. code-block:: yaml

       configuration:
           some_risk:
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
            The type and name of a risk, specified as "type.name". Type is singular.
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
        """Set up the risk component.

        Extend parent setup to register an exposure column for
        non-loglinear risk effects when applicable.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.
        """
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
        """Initialize an exposure column with the exposure pipeline values.

        Parameters
        ----------
        pop_data
            Metadata about the simulants being initialized.
        """
        exposure = self.get_exposure(pop_data.index)
        self.population_view.initialize(exposure)

    def on_time_step_prepare(self, event: Event) -> None:
        """Update the exposure column to equal the exposure pipeline values if there is a
        NonLogLinearRiskEffect component for this risk in the simulation.

        Parameters
        ----------
        event
            The event triggering the preparation.
        """
        if self.includes_non_loglinear_risk_effect:
            exposure = self.get_exposure(event.index)
            self.population_view.update(
                self.exposure_column_name,
                lambda _: exposure,
            )

    def get_exposure(self, index: pd.Index) -> pd.Series:
        """Get the exposure attribute and rename it to the internal exposure column name.

        HACK: This is effectively caching the exposure pipeline for use by other
        components. Specifically, :meth:`vivarium_public_health.risks.effect.NonLogLinearRiskEffect.get_relative_risk_source`
        needs the exposure values but calling that pipeline was very slow. By
        maintaining a cached copy of the exposure values in a private column, we
        can then request that corresponding "simple" pipeline from the population
        view instead which is significantly faster.

        Parameters
        ----------
        index
            The index of the population to update.
        """
        exposure = self.population_view.get(index, self.exposure_name)
        exposure.name = self.exposure_column_name
        return exposure
