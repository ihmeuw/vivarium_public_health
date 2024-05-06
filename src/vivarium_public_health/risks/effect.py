"""
==================
Risk Effect Models
==================

This module contains tools for modeling the relationship between risk
exposure models and disease models.

"""

from typing import Any, Callable, Dict, List, Tuple

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

    ###############
    # Properties #
    ##############

    @property
    def name(self) -> str:
        return self.get_name(self.risk, self.target)

    @staticmethod
    def get_name(risk: EntityString, target: TargetString) -> str:
        return f"risk_effect.{risk.name}_on_{target.name}"

    @property
    def configuration_defaults(self) -> Dict[str, Any]:
        """
        A dictionary containing the defaults for any configurations managed by
        this component.
        """
        return {
            self.name: {
                "data_sources": {
                    "relative_risk": "self::get_relative_risk_source",
                    "population_attributable_fraction": "self::get_population_attributable_fraction_source",
                },
                "distribution_args": {
                    "mean": None,
                    "se": None,
                    "log_mean": None,
                    "log_se": None,
                    "tau_squared": None,
                },
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
        relative_risk_data, rr_value_cols = self.get_relative_risk_source(builder)
        self.lookup_tables["relative_risk"] = self.build_lookup_table(
            builder, relative_risk_data, rr_value_cols
        )
        paf_data, paf_value_cols = self.get_population_attributable_fraction_source(builder)
        self.lookup_tables["population_attributable_fraction"] = self.build_lookup_table(
            builder, paf_data, paf_value_cols
        )

    def get_distribution_type(self, builder: Builder) -> str:
        return get_distribution_type(builder, self.risk)

    def get_risk_exposure(self, builder: Builder) -> Callable[[pd.Index], pd.Series]:
        return builder.value.get_value(self.exposure_pipeline_name)

    def get_relative_risk_source(self, builder: Builder) -> Tuple[pd.DataFrame, List[str]]:
        """
        Get the relative risk source for this risk effect model.

        Parameters
        ----------
        builder
            Interface to access simulation managers.

        Returns
        -------
        LookupTable
            A lookup table containing the relative risk data for this risk
            effect model.
        """
        return get_relative_risk_data(builder, self.risk, self.target)

    def get_population_attributable_fraction_source(
        self, builder: Builder
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Get the population attributable fraction source for this risk effect model.

        Parameters
        ----------
        builder
            Interface to access simulation managers.

        Returns
        -------
        LookupTable
            A lookup table containing the population attributable fraction data
            for this risk effect model.
        """
        return get_population_attributable_fraction_data(builder, self.risk, self.target)

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
