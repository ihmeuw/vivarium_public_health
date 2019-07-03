"""
===================
Risk Exposure Model
===================

This module contains tools for modeling categorical and continuous risk
exposure.

"""
import pandas as pd

from vivarium_public_health.utilities import EntityString
from vivarium_public_health.risks.data_transformations import get_distribution, get_exposure_post_processor


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
            "exposure": 'data',
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
        self.configuration_defaults = {f'{self.risk.name}': Risk.configuration_defaults['risk']}

    @property
    def name(self):
        return f'risk.{self.risk}'

    def setup(self, builder):
        self.exposure_distribution = self._get_distribution(builder)
        builder.components.add_components([self.exposure_distribution])

        self.randomness = builder.randomness.get_stream(f'initial_{self.risk.name}_propensity')

        self._propensity = pd.Series()
        self.propensity = builder.value.register_value_producer(f'{self.risk.name}.propensity',
                                                                source=lambda index: self._propensity[index])
        self.exposure = builder.value.register_value_producer(
            f'{self.risk.name}.exposure',
            source=self.get_current_exposure,
            preferred_post_processor=get_exposure_post_processor(builder, self.risk)
        )

        builder.population.initializes_simulants(self.on_initialize_simulants)

    def on_initialize_simulants(self, pop_data):
        self._propensity = self._propensity.append(self.randomness.get_draw(pop_data.index))

    def get_current_exposure(self, index):
        propensity = self.propensity(index)

        return pd.Series(self.exposure_distribution.ppf(propensity), index=index)

    def _get_distribution(self, builder):
        return get_distribution(builder, self.risk)

    def __repr__(self):
        return f"Risk({self.risk})"

