"""
==========================
CausalFactor Effect Models
==========================

This module contains tools for modeling the relationship between causal factor
exposure models and the models they affect.

"""

from abc import ABC
from collections.abc import Callable
from importlib import import_module
from typing import Any

import numpy as np
import pandas as pd
from layered_config_tree import ConfigurationError
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.lookup import LookupTable
from vivarium.types import LookupTableData

from vivarium_public_health.causal_factor.calibration_constant import (
    get_calibration_constant_pipeline_name,
)
from vivarium_public_health.causal_factor.distributions import DichotomousDistribution
from vivarium_public_health.causal_factor.exposure import CausalFactor
from vivarium_public_health.causal_factor.utilities import (
    load_exposure_data,
    pivot_categorical,
)
from vivarium_public_health.utilities import EntityString, TargetString


class CausalFactorEffect(Component, ABC):
    """A component to model the effect of a causal factor on an affected entity's target measure.

    This component can source data either from builder.data or from parameters
    supplied in the configuration.

    For a causal factor named 'causal_factor' that affects 'affected_target', the configuration
    would look like:

    .. code-block:: yaml

       configuration:
            causal_factor_effect.causal_factor_name_on_affected_target:
               exposure_parameters: 2
               incidence_rate: 10

    """

    EXPOSURE_CLASS = CausalFactor

    ##############
    # Properties #
    ##############

    @property
    def name(self) -> str:
        """The name of this causal factor effect component."""
        return self.get_name(self.causal_factor, self.target)

    @staticmethod
    def get_name(causal_factor: EntityString, target: TargetString) -> str:
        """Return the component name for a causal factor and target pair."""
        return f"causal_factor_effect.{causal_factor.name}_on_{target}"

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        """Default configuration values for this component.

        Configuration structure::

            {causal_factor_effect_name}:
                data_sources:
                    relative_risk:
                        Source for relative risk data. Default is the artifact
                        key ``{causal_factor}.relative_risk``. Can also be:
                        - A scalar value (e.g., ``1.5``)
                        - A scipy.stats distribution name (e.g., ``"uniform"``)
                          with parameters in ``data_source_parameters``
                    population_attributable_fraction:
                        Source for PAF data. Default is the artifact key
                        ``{causal_factor}.population_attributable_fraction``. Used to
                        adjust the target measure to account for the portion
                        attributable to this causal factor.
                data_source_parameters:
                    relative_risk: dict
                        Parameters for scipy.stats distributions when using
                        a distribution name as the ``relative_risk`` source.
                        For example, ``{"loc": 1.0, "scale": 0.5}`` for a
                        uniform distribution.
        """
        return {
            self.name: {
                "data_sources": {
                    "relative_risk": f"{self.causal_factor}.relative_risk",
                    "population_attributable_fraction": f"{self.causal_factor}.population_attributable_fraction",
                },
                "data_source_parameters": {
                    "relative_risk": {},
                },
            }
        }

    @property
    def is_exposure_categorical(self) -> bool:
        """Whether the exposure distribution is categorical."""
        return self._exposure_distribution_type in [
            "dichotomous",
            "ordered_polytomous",
            "unordered_polytomous",
        ]

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, causal_factor: str, target: str):
        """

        Parameters
        ----------
        causal_factor
            Type and name of causal factor, supplied in the form
            "causal_factor_type.causal_factor_name" where causal_factor_type should be singular (e.g.,
            risk_factor instead of risk_factors).
        target
            Type, name, and target measure of entity to be affected by causal factor,
            supplied in the form "entity_type.entity_name.measure"
            where entity_type should be singular (e.g., cause instead of causes).
        """
        super().__init__()
        self.causal_factor = EntityString(causal_factor)
        self.target = TargetString(target)

        self._exposure_distribution_type = None

        self.exposure_name = f"{self.causal_factor.name}.exposure"
        self.target_name = f"{self.target.name}.{self.target.measure}"
        self.relative_risk_name = (
            f"{self.causal_factor.name}_on_{self.target_name}.relative_risk"
        )

    def setup(self, builder: Builder) -> None:
        """Set up the causal factor effect component.

        Load distribution type and PAF data, define relative risk source,
        build relative risk lookup tables, register relative risk pipeline,
        and register target and calibration constant modifiers.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.
        """
        self.causal_factor_exposure_component = self._get_causal_factor_exposure_component(
            builder
        )
        self._exposure_distribution_type = self.get_distribution_type(builder)
        self.relative_risk_table = self.build_rr_lookup_table(builder)
        self.paf_data = self.get_calibration_constant_data(builder)

        self._relative_risk_source = self.get_relative_risk_source(builder)
        self.register_relative_risk_pipeline(builder)

        self.register_target_modifier(builder)
        self.register_calibration_constant_modifier(builder)

    #################
    # Setup methods #
    #################

    def build_rr_lookup_table(self, builder: Builder) -> LookupTable:
        """Build a lookup table for relative risk data.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.

        Returns
        -------
            A lookup table of relative risk values.
        """
        rr_data = self.load_relative_risk(builder)
        rr_value_cols = None
        if self.is_exposure_categorical:
            rr_data, rr_value_cols = self.process_categorical_data(builder, rr_data)
        return self.build_lookup_table(
            builder, "relative_risk", data_source=rr_data, value_columns=rr_value_cols
        )

    def get_calibration_constant_data(self, builder: Builder) -> LookupTableData:
        """Load calibration constant (PAF) data for this effect.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.

        Returns
        -------
            The calibration constant data.
        """
        return self.get_filtered_data(
            builder, self.configuration.data_sources.population_attributable_fraction
        )

    def get_distribution_type(self, builder: Builder) -> str:
        """Get the distribution type for the causal factor from the configuration."""
        return (
            self.causal_factor_exposure_component.distribution_type
            or self.causal_factor_exposure_component.get_distribution_type(builder)
        )

    def load_relative_risk(
        self,
        builder: Builder,
        configuration=None,
    ) -> str | float | pd.DataFrame:
        """Load relative risk data from the configuration.

        Attempt to interpret the configured source as a scipy.stats
        distribution name; if that fails, load it as artifact data.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.
        configuration
            Optional configuration override. If ``None``, use
            ``self.configuration``.

        Returns
        -------
            The relative risk data.

        Raises
        ------
        ConfigurationError
            If the distribution parameters are invalid.
        """
        if configuration is None:
            configuration = self.configuration

        rr_source = configuration.data_sources.relative_risk
        rr_dist_parameters = configuration.data_source_parameters.relative_risk.to_dict()

        if isinstance(rr_source, str):
            try:
                distribution = getattr(import_module("scipy.stats"), rr_source)
                rng = np.random.default_rng(builder.randomness.get_seed(self.name))
                rr_data = distribution(**rr_dist_parameters).ppf(rng.random())
            except AttributeError:
                rr_data = self.get_filtered_data(builder, rr_source)
            except TypeError:
                raise ConfigurationError(
                    f"Parameters {rr_dist_parameters} are not valid for distribution {rr_source}."
                )
        else:
            rr_data = self.get_filtered_data(builder, rr_source)
        return rr_data

    def get_filtered_data(
        self, builder: Builder, data_source: str | float | pd.DataFrame
    ) -> float | pd.DataFrame:
        """Load data and filter to the target entity and measure.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.
        data_source
            The data source identifier, scalar, or DataFrame.

        Returns
        -------
            The filtered data.
        """
        data = self.get_data(builder, data_source)

        if isinstance(data, pd.DataFrame):
            # filter data to only include the target entity and measure
            correct_target_mask = True
            columns_to_drop = []
            if "affected_entity" in data.columns:
                correct_target_mask &= data["affected_entity"] == self.target.name
                columns_to_drop.append("affected_entity")
            if "affected_measure" in data.columns:
                correct_target_mask &= data["affected_measure"] == self.target.measure
                columns_to_drop.append("affected_measure")
            data = data[correct_target_mask].drop(columns=columns_to_drop)
        return data

    def process_categorical_data(
        self, builder: Builder, rr_data: str | float | pd.DataFrame
    ) -> tuple[str | float | pd.DataFrame, list[str]]:
        """Process relative risk data for categorical exposures.

        For scalar RR data with a dichotomous distribution, construct a
        DataFrame with exposed/unexposed categories. Pivot the data to
        wide format for use in a lookup table.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.
        rr_data
            The relative risk data.

        Returns
        -------
            A tuple of the pivoted RR data and the list of value column
            names.

        Raises
        ------
        ValueError
            If scalar RR data is provided with a non-dichotomous
            distribution.
        """
        if not isinstance(rr_data, pd.DataFrame):
            exposure_distribution = (
                self.causal_factor_exposure_component.exposure_distribution
            )
            if not isinstance(exposure_distribution, DichotomousDistribution):
                raise ValueError(
                    f"Relative risk data for categorical exposure must be a DataFrame unless the "
                    f"exposure distribution is dichotomous. Found type {type(rr_data)} with "
                    f"exposure distribution type {exposure_distribution.distribution_type}."
                )
            cat1 = builder.data.load("population.demographic_dimensions")
            cat1["parameter"] = exposure_distribution.exposed
            cat1["value"] = rr_data
            cat2 = cat1.copy()
            cat2["parameter"] = exposure_distribution.unexposed
            cat2["value"] = 1
            rr_data = pd.concat([cat1, cat2], ignore_index=True)
        if "parameter" in rr_data.index.names:
            rr_data = rr_data.reset_index("parameter")

        exposure_distribution = self.causal_factor_exposure_component.exposure_distribution
        if isinstance(exposure_distribution, DichotomousDistribution):
            rr_data = exposure_distribution.rename_deprecated_categories(rr_data)

        rr_value_cols = list(rr_data["parameter"].unique())
        rr_data = pivot_categorical(rr_data, "parameter")
        return rr_data, rr_value_cols

    # todo currently this isn't being called. we need to properly set rrs if
    #  the exposure has been rebinned
    def rebin_relative_risk_data(
        self, builder, relative_risk_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Rebin relative risk data.

        When the polytomous risk is rebinned, matching relative risk needs to be rebinned.
        After rebinning, rr for both exposed and unexposed categories should be the weighted sum of relative risk
        of the component categories where weights are relative proportions of exposure of those categories.
        For example, if cat1, cat2, cat3 are exposed categories and cat4 is unexposed with exposure [0.1,0.2,0.3,0.4],
        for the matching rr = [rr1, rr2, rr3, 1], rebinned rr for the rebinned cat1 should be:
        (0.1 *rr1 + 0.2 * rr2 + 0.3* rr3) / (0.1+0.2+0.3)
        """
        if not self.causal_factor in builder.configuration.to_dict():
            return relative_risk_data

        rebin_exposed_categories = set(
            builder.configuration[self.causal_factor]["rebinned_exposed"]
        )

        if rebin_exposed_categories:
            # todo make sure this works
            exposure_data = load_exposure_data(builder, self.causal_factor)
            relative_risk_data = self._rebin_relative_risk_data(
                relative_risk_data, exposure_data, rebin_exposed_categories
            )

        return relative_risk_data

    def _rebin_relative_risk_data(
        self,
        relative_risk_data: pd.DataFrame,
        exposure_data: pd.DataFrame,
        rebin_exposed_categories: set,
    ) -> pd.DataFrame:
        """Compute exposure-weighted relative risks for rebinned categories."""
        cols = list(exposure_data.columns.difference(["value"]))

        relative_risk_data = relative_risk_data.merge(exposure_data, on=cols)
        relative_risk_data["value_x"] = relative_risk_data.value_x.multiply(
            relative_risk_data.value_y
        )
        relative_risk_data.parameter = relative_risk_data["parameter"].map(
            lambda p: "cat1" if p in rebin_exposed_categories else "cat2"
        )
        relative_risk_data = relative_risk_data.groupby(cols).sum().reset_index()
        relative_risk_data["value"] = relative_risk_data.value_x.divide(
            relative_risk_data.value_y
        ).fillna(0)
        return relative_risk_data.drop(columns=["value_x", "value_y"])

    def get_relative_risk_source(self, builder: Builder) -> Callable[[pd.Index], pd.Series]:
        """Build a callable that computes relative risk from exposure.

        For continuous exposures, use TMRED-based log-linear scaling.
        For categorical exposures, look up the RR for each simulant's
        exposure category.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.

        Returns
        -------
            A callable that accepts a simulant index and returns
            relative risk values.
        """

        if not self.is_exposure_categorical:
            tmred = builder.data.load(f"{self.causal_factor}.tmred")
            tmrel = 0.5 * (tmred["min"] + tmred["max"])
            scale = builder.data.load(f"{self.causal_factor}.relative_risk_scalar")

            def generate_relative_risk(index: pd.Index) -> pd.Series:
                rr = self.relative_risk_table(index)
                exposure = self.population_view.get_attributes(index, self.exposure_name)
                relative_risk = np.maximum(rr.values ** ((exposure - tmrel) / scale), 1)
                return relative_risk

        else:
            index_columns = ["index", self.causal_factor.name]

            def generate_relative_risk(index: pd.Index) -> pd.Series:
                rr = self.relative_risk_table(index)
                exposure = self.population_view.get_attributes(
                    index, self.exposure_name
                ).reset_index()
                exposure.columns = index_columns
                exposure = exposure.set_index(index_columns)

                relative_risk = rr.stack().reset_index()
                relative_risk.columns = index_columns + ["value"]
                relative_risk = relative_risk.set_index(index_columns)

                effect = relative_risk.loc[exposure.index, "value"].droplevel(
                    self.causal_factor.name
                )
                return effect

        return generate_relative_risk

    def register_relative_risk_pipeline(self, builder: Builder) -> None:
        """Register the relative risk pipeline with the simulation.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.
        """
        builder.value.register_attribute_producer(
            self.relative_risk_name,
            self._relative_risk_source,
            required_resources=[self.exposure_name],
        )

    def register_target_modifier(self, builder: Builder) -> None:
        """Register the relative risk as a modifier on the target pipeline.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.
        """
        builder.value.register_attribute_modifier(
            self.target_name, modifier=self.relative_risk_name
        )

    def register_calibration_constant_modifier(self, builder: Builder) -> None:
        """Register the PAF data as a modifier on the calibration constant pipeline.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.
        """
        builder.value.register_value_modifier(
            get_calibration_constant_pipeline_name(self.target_name),
            modifier=lambda: self.paf_data,
        )

    ##################
    # Helper methods #
    ##################

    def _get_causal_factor_exposure_component(self, builder: Builder) -> CausalFactor:
        """Retrieve effect component and validate that it is compatible with the
        causal factor exposure.
        """
        causal_factor_exposure_component = builder.components.get_component(
            self.causal_factor
        )
        if not isinstance(causal_factor_exposure_component, self.EXPOSURE_CLASS):
            raise ValueError(
                f"{self.__class__.__name__} model {self.name} requires a {self.EXPOSURE_CLASS.__name__} component named {self.causal_factor}"
            )
        return causal_factor_exposure_component
