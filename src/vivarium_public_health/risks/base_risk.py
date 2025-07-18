"""
===================
Risk Exposure Model
===================

This module contains tools for modeling categorical and continuous risk
exposure.

"""
from typing import NamedTuple

from vivarium.framework.engine import Builder

from vivarium_public_health.exposure import Exposure


class Risk(Exposure):
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

    @property
    def measure_name(self) -> str:
        """The measure of the risk exposure."""
        return "exposure"

    @property
    def dichotomous_exposure_category_names(self) -> NamedTuple:
        """The name of the exposed category for this intervention."""

        class __Categories(NamedTuple):
            exposed: str = "exposed"
            unexposed: str = "unexposed"

        categories = __Categories()
        return categories

    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        # We want to set this to True if there is a non-loglinear risk effect
        # on this risk instance
        self.create_exposure_column = bool(
            [
                component
                for component in builder.components.list_components()
                if component.startswith(f"non_log_linear_risk_effect.{self.entity.name}_on_")
            ]
        )
