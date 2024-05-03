"""
==================
Risk Effect Models
==================

This module contains tools for modeling the relationship between risk
exposure models and disease models.

"""

from typing import Any, Callable, Dict

import numpy as np
import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder

from vivarium_public_health.risks.data_transformations import (
    get_distribution_type,
    get_population_attributable_fraction_data,
    get_relative_risk_data,
)
from vivarium_public_health.utilities import (
    EntityString,
    TargetString,
    get_lookup_columns,
)


class RiskEffect(Component):
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

    CONFIGURATION_DEFAULTS = {
        "effect_of_risk_on_target": {
            "measure": {
                "relative_risk": None,
                "mean": None,
                "se": None,
                "log_mean": None,
                "log_se": None,
                "tau_squared": None,
            }
        }
    }

    ##############
    # Properties #
    ##############

    @property
    def configuration_defaults(self) -> Dict[str, Any]:
        """
        A dictionary containing the defaults for any configurations managed by
        this component.
        """
        return {
            f"effect_of_{self.risk.name}_on_{self.target.name}": {
                self.target.measure: self.CONFIGURATION_DEFAULTS["effect_of_risk_on_target"][
                    "measure"
                ]
            }
        }

    #####################
    # Lifecycle methods #
    #####################

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
        super().__init__()
        self.risk = EntityString(risk)
        self.target = TargetString(target)

        self.exposure_pipeline_name = f"{self.risk.name}.exposure"
        self.target_pipeline_name = f"{self.target.name}.{self.target.measure}"
        self.target_paf_pipeline_name = f"{self.target_pipeline_name}.paf"

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.exposure_distribution_type = self.get_distribution_type(builder)
        self.exposure = self.get_risk_exposure(builder)

        self.target_modifier = self.get_target_modifier(builder)

        self.register_target_modifier(builder)
        self.register_paf_modifier(builder)

    #################
    # Setup methods #
    #################

    def build_all_lookup_tables(self, builder: Builder) -> None:
        paf_data = get_population_attributable_fraction_data(builder, self.risk, self.target)
        self.lookup_tables["population_attributable_fraction"] = self.build_lookup_table(
            builder, paf_data
        )
        relative_risk_data = get_relative_risk_data(builder, self.risk, self.target)
        self.lookup_tables["relative_risk"] = self.build_lookup_table(
            builder, relative_risk_data
        )

    def get_distribution_type(self, builder: Builder) -> str:
        return get_distribution_type(builder, self.risk)

    def get_risk_exposure(self, builder: Builder) -> Callable[[pd.Index], pd.Series]:
        return builder.value.get_value(self.exposure_pipeline_name)

    def get_target_modifier(
        self, builder: Builder
    ) -> Callable[[pd.Index, pd.Series], pd.Series]:
        if self.exposure_distribution_type in ["normal", "lognormal", "ensemble"]:
            tmred = builder.data.load(f"{self.risk}.tmred")
            tmrel = 0.5 * (tmred["min"] + tmred["max"])
            scale = builder.data.load(f"{self.risk}.relative_risk_scalar")

            def adjust_target(index: pd.Index, target: pd.Series) -> pd.Series:
                rr = self.lookup_tables["relative_risk"](index)
                exposure = self.exposure(index)
                relative_risk = np.maximum(rr.values ** ((exposure - tmrel) / scale), 1)
                return target * relative_risk

        else:
            index_columns = ["index", self.risk.name]

            def adjust_target(index: pd.Index, target: pd.Series) -> pd.Series:
                rr = self.lookup_tables["relative_risk"](index)
                exposure = self.exposure(index).reset_index()
                exposure.columns = index_columns
                exposure = exposure.set_index(index_columns)

                relative_risk = rr.stack().reset_index()
                relative_risk.columns = index_columns + ["value"]
                relative_risk = relative_risk.set_index(index_columns)

                effect = relative_risk.loc[exposure.index, "value"].droplevel(self.risk.name)
                affected_rates = target * effect
                return affected_rates

        return adjust_target

    def register_target_modifier(self, builder: Builder) -> None:
        builder.value.register_value_modifier(
            self.target_pipeline_name,
            modifier=self.target_modifier,
            requires_values=[f"{self.risk.name}.exposure"],
        )

    def register_paf_modifier(self, builder: Builder) -> None:
        required_columns = get_lookup_columns(
            [self.lookup_tables["population_attributable_fraction"]]
        )
        builder.value.register_value_modifier(
            self.target_paf_pipeline_name,
            modifier=self.lookup_tables["population_attributable_fraction"],
            requires_columns=required_columns,
        )
