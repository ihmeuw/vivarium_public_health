import pandas as pd

from vivarium_public_health.risks import get_distribution
from vivarium_public_health.risks.data_transformation import build_exp_data_from_config, split_risk_from_type


class Risk:
    """A model for a risk factor defined by either a continuous or a categorical
    value. For example,
    (1) high systolic blood pressure as a risk where the SBP is not dichotomized
        into hypotension and normal but is treated as the actual SBP measurement.
    (2) smoking as two categories: current smoker and non-smoker.

    This component can source data either from builder.data or from parameters
    supplied in the configuration. If data is derived from the configuration, it
    must be an integer or float expressing the desired exposure level or a
    covariate name that is intended to be used as
    a proxy. For example, for a risk named "risk", the configuration could look
    like this:

    (1) configuration:
            risk:
                exposure: 1.0
    (2) configuration:
            risk:
                exposure: proxy_covariate
    """

    configuration_defaults = {
        "risk": {
            "exposure": 'data',
            "distribution": 'dichotomous'
        }
    }

    def __init__(self, risk: str):
        """

        Parameters
        ----------
        risk :
            the type and name of a risk, specified as "type.name". Type is singular.

        """
        self._risk_type, self._risk = split_risk_from_type(risk)
        self.configuration_defaults = {f'{self._risk}': Risk.configuration_defaults['risk']}

    def setup(self, builder):
        self.exposure_distribution = self._get_distribution(builder)
        builder.components.add_components([self.exposure_distribution])
        self.randomness = builder.randomness.get_stream(f'initial_{self._risk}_propensity')
        self._propensity = pd.Series()
        self.propensity = builder.value.register_value_producer(f'{self._risk}.propensity',
                                                                source=lambda index: self._propensity[index])
        self.exposure = builder.value.register_value_producer(f'{self._risk}.exposure',
                                                              source=self.get_current_exposure)
        builder.population.initializes_simulants(self.on_initialize_simulants)

    def _get_distribution(self, builder):
        """A wrapper to isolate builder from setup"""
        kwargs = {}
        if builder.configuration[self._risk]['exposure'] != 'data':
            if builder.configuration[self._risk]['distribution'] != "dichotomous":
                raise NotImplementedError("A dichotomous distribution is currently the only supported distribution for "
                                          "a risk with data supplied via configuration")

            exposure_data = build_exp_data_from_config(builder, self._risk)
            distribution_type = builder.configuration[self._risk]['distribution']

        else:
            distribution_type = builder.data.load(f"{self._risk_type}.{self._risk}.distribution")
            exposure_data = builder.data.load(f"{self._risk_type}.{self._risk}.exposure")

            if distribution_type == "polytomous":
                kwargs['configuration'] = builder.configuration
            elif distribution_type in ['normal', 'lognormal', 'ensemble']:
                kwargs['exposure_standard_deviation'] = builder.data.load(f"{self._risk_type}.{self._risk}.exposure_standard_deviation")
                if distribution_type == "ensemble":
                    kwargs['weights'] = builder.data.load(f'risk_factor.{self._risk}.ensemble_weights')

        return get_distribution(self._risk, distribution_type, exposure_data, **kwargs)

    def on_initialize_simulants(self, pop_data):
        self._propensity = self._propensity.append(self.randomness.get_draw(pop_data.index))

    def get_current_exposure(self, index):
        propensity = self.propensity(index)
        return self.exposure_distribution.ppf(propensity)

    def __repr__(self):
        return f"Risk(_risk_type= {self._risk_type}, _risk= {self._risk})"
