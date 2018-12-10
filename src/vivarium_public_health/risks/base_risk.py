import pandas as pd

from vivarium_public_health.risks import get_distribution
from vivarium_public_health.risks.data_transformation import build_exp_data_from_config


class Risk:
    """A model for a risk factor defined by either a continuous or a categorical value. For example,
    (1) high systolic blood pressure as a risk where the SBP is not dichotomized
        into hypotension and normal but is treated as the actual SBP measurement.
    (2) smoking as two categories: current smoker and non-smoker.


       Parameters
       ----------
       risk_type : str
           'risk_factor'
       risk_name : str
           The name of a risk
       """
    configuration_defaults = {}

    def __init__(self, risk_type: str, risk_name: str):
        self._risk_type, self._risk = risk_type, risk_name

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

        distribution_type = builder.data.load(f"{self._risk_type}.{self._risk}.distribution")
        exposure_data = builder.data.load(f"{self._risk_type}.{self._risk}.exposure")

        kwargs = {}
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


class DummyRisk(Risk):
    """A model for a risk factor defined by either a continuous or a categorical value. For example,
    (1) high systolic blood pressure as a risk where the SBP is not dichotomized
        into hypotension and normal but is treated as the actual SBP measurement.
    (2) smoking as two categories: current smoker and non-smoker.

    The difference between this component and the Risk component is the source of the data. This Dummy
    Risk derives data from the configuration file itself. This data can be an integer/float of the
    desired exposure level, or a string covariate name that is intended to be used as a proxy.

    Parameters
    ----------
    risk_type : str
       'risk_factor'
    risk_name : str
       The name of a risk
    """

    configuration_defaults = {
        "dummy_risk": {
            "exposure": "1.0",
            "distribution": "dichotomous"
        }
    }

    def __init__(self, risk_type: str, risk_name: str):
        super().__init__(risk_type, risk_name)
        self._risk_type, self._risk = risk_type, risk_name
        self.configuration_defaults = {f'{self._risk}': DummyRisk.configuration_defaults['dummy_risk']}

    def _get_distribution(self, builder):

        if builder.configuration[self._risk]['distribution'] != "dichotomous":
            raise ValueError("A Dummy Risk must have a dichotomous distribution.")

        exposure_data = build_exp_data_from_config(builder, self._risk)
        distribution_type = builder.configuration[self._risk]['distribution']

        return get_distribution(self._risk, distribution_type, exposure_data)

    def __repr__(self):
        return f"DummyRisk(_risk_type={self._risk_type}, _risk={self._risk}"
