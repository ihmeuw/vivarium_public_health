"""
==================
Risk Effect Models
==================

This module contains tools for modeling the relationship between risk
exposure models and disease models.

"""

from typing import Callable, Dict

import pandas as pd

from vivarium.framework.engine import Builder
from vivarium.framework.lookup import LookupTable

from vivarium_public_health.risks.data_transformations import (get_relative_risk_data,
                                                               get_population_attributable_fraction_data,
                                                               get_exposure_effect)
from vivarium_public_health.utilities import EntityString, TargetString


class RiskEffect:
    """A component to model the impact of a risk factor on the target rate of
    some affected entity. This component can source data either from
    builder.data or from parameters supplied in the configuration.
    For a risk named 'risk' that affects 'affected_risk' and 'affected_cause',
    the configuration would look like:

    .. code-block:: yaml

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
        self.configuration_defaults = self.get_configuration_defaults()

    def __repr__(self):
        return f"RiskEffect(risk={self.risk}, target={self.target})"

    ##########################
    # Initialization methods #
    ##########################

    def get_configuration_defaults(self) -> Dict[str, Dict]:
        return {
            f'effect_of_{self.risk.name}_on_{self.target.name}': {
                self.target.measure: RiskEffect.configuration_defaults['effect_of_risk_on_target']['measure']
            }
        }

    ##############
    # Properties #
    ##############

    @property
    def name(self):
        return f'risk_effect.{self.risk}.{self.target}'

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder):
        self.relative_risk = self.get_relative_risk_lookup_table(builder)
        self.population_attributable_fraction = self.get_population_attributable_fraction_lookup_table(builder)
        self.target_modifier = self.get_target_modifier(builder)

        self.register_target_modifier(builder)
        self.register_paf_modifier(builder)

    def get_relative_risk_lookup_table(self, builder: Builder) -> LookupTable:
        relative_risk_data = get_relative_risk_data(builder, self.risk, self.target)
        return builder.lookup.build_table(relative_risk_data,
                                          key_columns=['sex'],
                                          parameter_columns=['age', 'year'])

    def get_population_attributable_fraction_lookup_table(self, builder: Builder) -> LookupTable:
        paf_data = get_population_attributable_fraction_data(builder, self.risk, self.target)
        return builder.lookup.build_table(paf_data,
                                          key_columns=['sex'],
                                          parameter_columns=['age', 'year'])

    def get_target_modifier(self, builder: Builder) -> Callable[[pd.Index, pd.Series], pd.Series]:
        exposure_effect = get_exposure_effect(builder, self.risk)

        def adjust_target(index: pd.Index, target: pd.Series) -> pd.Series:
            return exposure_effect(target, self.relative_risk(index))

        return adjust_target

    def register_target_modifier(self, builder: Builder) -> None:
        builder.value.register_value_modifier(f'{self.target.name}.{self.target.measure}',
                                              modifier=self.target_modifier,
                                              requires_values=[f'{self.risk.name}.exposure'],
                                              requires_columns=['age', 'sex'])

    def register_paf_modifier(self, builder: Builder) -> None:
        builder.value.register_value_modifier(f'{self.target.name}.{self.target.measure}.paf',
                                              modifier=self.population_attributable_fraction,
                                              requires_columns=['age', 'sex'])
