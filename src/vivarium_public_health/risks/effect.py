
from vivarium_public_health.utilities import EntityString, TargetString
from .data_transformations import (get_relative_risk_data, get_population_attributable_fraction_data,
                                   get_exposure_effect)


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
        'effect_of_risk_on_target': {
            'measure': {
                'relative_risk': None,
                'mean': None,
                'se': None,
                'log_mean': None,
                'log_se': None,
                'tau_squared': None
            }
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
        """
        self.risk = EntityString(risk)
        self.target = TargetString(target)
        self.configuration_defaults = {
            f'effect_of_{self.risk.name}_on_{self.target.name}': {
                self.target.measure: RiskEffect.configuration_defaults['effect_of_risk_on_target']['measure']
            }
        }

    @property
    def name(self):
        return f'risk_effect.{self.risk}.{self.target}'

    def setup(self, builder):
        self.randomness = builder.randomness.get_stream(f'effect_of_{self.risk.name}_on_{self.target.name}')
        self.relative_risk = builder.lookup.build_table(get_relative_risk_data(builder, self.risk,
                                                                               self.target, self.randomness))
        self.population_attributable_fraction = builder.lookup.build_table(
            get_population_attributable_fraction_data(builder, self.risk, self.target, self.randomness)
        )

        self.exposure_effect = get_exposure_effect(builder, self.risk)

        builder.value.register_value_modifier(f'{self.target.name}.{self.target.measure}', modifier=self.adjust_target)
        builder.value.register_value_modifier(f'{self.target.name}.{self.target.measure}.paf',
                                              modifier=self.population_attributable_fraction)

    def adjust_target(self, index, target):
        return self.exposure_effect(target, self.relative_risk(index))

    def __repr__(self):
        return f"RiskEffect(risk={self.risk}, target={self.target})"
