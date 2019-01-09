import pandas as pd

from vivarium_public_health.risks import get_distribution
from vivarium_public_health.risks.data_transformation import get_exposure_data, RiskString


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
            "distribution": 'data',
            "rebin": {},
        }
    }

    def __init__(self, risk: str):
        """

        Parameters
        ----------
        risk :
            the type and name of a risk, specified as "type.name". Type is singular.

        """
        self.risk = RiskString(risk)
        self.configuration_defaults = {f'{self.risk.name}': Risk.configuration_defaults['risk']}

    def setup(self, builder):
        self._validate_data_source(builder)
        self.exposure_distribution = self._get_distribution(builder)
        builder.components.add_components([self.exposure_distribution])
        self.randomness = builder.randomness.get_stream(f'initial_{self.risk.name}_propensity')
        self._propensity = pd.Series()
        self.propensity = builder.value.register_value_producer(f'{self.risk.name}.propensity',
                                                                source=lambda index: self._propensity[index])
        self.exposure = builder.value.register_value_producer(f'{self.risk.name}.exposure',
                                                              source=self.get_current_exposure)
        builder.population.initializes_simulants(self.on_initialize_simulants)

    def on_initialize_simulants(self, pop_data):
        self._propensity = self._propensity.append(self.randomness.get_draw(pop_data.index))

    def get_current_exposure(self, index):
        propensity = self.propensity(index)
        return self.exposure_distribution.ppf(propensity)

    def _get_distribution(self, builder):
        risk_config = builder.configuration[self.risk.name]
        distribution_type = risk_config['distribution']
        if distribution_type == 'data':
            distribution_type = builder.data.load(f'{self.risk}.distribution')

        args = {'risk': self.risk.name,
                'distribution_type': distribution_type,
                'exposure': get_exposure_data(builder, self.risk),
                'rebin': risk_config['rebin'],
                'exposure_standard_deviation': None,
                'weights': None}

        if distribution_type in ['normal', 'lognormal', 'ensemble']:
            args['exposure_standard_deviation'] = builder.data.load(f"{self.risk}.exposure_standard_deviation")
        if distribution_type == 'ensemble':
            args['weights'] = builder.data.load(f'{self.risk}.ensemble_weights')

        return get_distribution(**args)

    def _validate_data_source(self, builder):
        risk_config = builder.configuration[self.risk.name]
        if risk_config['exposure'] != 'data' and risk_config['distribution'] != 'dichotomous':
            raise ValueError('Parameterized risk components are only valid for dichotomous risks.')
        if isinstance(risk_config['exposure'], (int, float)) and not 0 <= risk_config['exposure'] <= 1:
            raise ValueError(f"Exposure should be in the range [0, 1]")

    def __repr__(self):
        return f"Risk({self.risk})"



