import numpy as np

from .data_transformation import (get_relative_risk_data, get_population_attributable_fraction_data,
                                  RiskString, TargetString, pivot_categorical)


class RiskEffect:
    """A component to model the impact of a risk factor on the target rate of
    some affected entity. This component can source data either from
    builder.data or from parameters supplied in the configuration.

    For a risk named 'risk' that affects 'affected_risk' and 'affected_cause',
    the configuration would look like:

    configuration:
        effect_of_risk_on_affected_risk:
            exposure_parameters: 2
            incidence_rate: 10
    """

    configuration_defaults = {
        'effect_of_risk_on_entity': {
            'incidence_rate': 'data',
            'exposure_parameters': 'data',
            'excess_mortality': 'data',
        }
    }

    def __init__(self, risk: str, target: str):
        """

        Parameters
        ----------
        risk :
            Type and name of risk factor, supplied in the form
            "risk_type.risk_name" where risk_type should be singular (e.g.,
            risk_factor instead of risk_factors).
        target :
            Type, name, and target rate of entity to be affected by risk factor,
            supplied in the form "entity_type.entity_name.measure"
            where entity_type should be singular (e.g., cause instead of causes).
        get_data_functions :
            Optional mapping of measure name to function to retrieve paf and rr
            data instead of reading from builder.data.

        """
        self.risk = RiskString(risk)
        self.target = TargetString(target)
        self.configuration_defaults = {f'effect_of_{self.risk.name}_on_{self.target.name}':
                                       RiskEffect.configuration_defaults['effect_of_risk_on_entity']}

    def setup(self, builder):
        self._check_valid_data_sources(builder)
        rr_data = get_relative_risk_data(builder, self.risk, self.target)
        rr_data = pivot_categorical(rr_data)
        self.relative_risk = builder.lookup.build_table(rr_data)
        paf_data = get_population_attributable_fraction_data(builder, self.risk, self.target)
        self.population_attributable_fraction = builder.lookup.build_table(paf_data)

        self.exposure_effect = self.get_exposure_effect(builder, self.risk)

        builder.value.register_value_modifier(f'{self.target.name}.{self.target.measure}', modifier=self.adjust_target)
        builder.value.register_value_modifier(f'{self.target.name}.{self.target.measure}.paf',
                                              modifier=self.population_attributable_fraction)

    def adjust_target(self, index, target):
        return self.exposure_effect(target, self.relative_risk(index))

    def _check_valid_data_sources(self, builder):
        risk_config = builder.configuration[self.risk.name]
        exposure_source = risk_config['exposure']
        rr_source = builder.configuration[f'effect_of_{self.risk.name}_on_{self.target.name}'][self.target.measure]
        if (exposure_source != 'data' or rr_source != 'data') and risk_config['distribution'] != 'dichotomous':
            raise ValueError('Parameterized risk components are only valid for dichotomous risks.')
        if isinstance(rr_source, (int, float)) and not 1 <= risk_config['exposure'] <= 100:
            raise ValueError(f"Relative risk should be in the range [1, 100]")

    @staticmethod
    def get_exposure_effect(builder, risk: RiskString):
        risk_config = builder.configuration[risk]
        distribution_type = risk_config['distribution']
        if distribution_type == 'data':
            distribution_type = builder.data.load(f'{risk}.distribution')

        risk_exposure = builder.value.get_value(f'{risk.name}.exposure')

        if distribution_type in ['normal', 'lognormal', 'ensemble']:
            raise NotImplementedError()
            # tmred = builder.data.load(f"{risk_type}.{risk}.tmred")
            # tmrel = 0.5 * (tmred["min"] + tmred["max"])
            # exposure_parameters = builder.data.load(f"{risk_type}.{risk}.exposure_parameters")
            # max_exposure = exposure_parameters["max_rr"]
            # scale = exposure_parameters["scale"]
            #
            # def exposure_effect(rates, rr):
            #     exposure = np.minimum(risk_exposure(rr.index), max_exposure)
            #     relative_risk = np.maximum(rr.values ** ((exposure - tmrel) / scale), 1)
            #     return rates * relative_risk
        else:

            def exposure_effect(rates, rr):
                exposure = risk_exposure(rr.index)
                return rates * (rr.lookup(exposure.index, exposure))

        return exposure_effect
