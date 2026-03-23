"""
==================
Risk Effect Models
==================

This module contains tools for modeling the relationship between risk
exposure models and disease models.

"""

from collections.abc import Callable
from importlib import import_module
from typing import Any

import numpy as np
import pandas as pd
import scipy
from layered_config_tree import ConfigurationError, LayeredConfigTree
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.lookup import LookupTable
from vivarium.types import LookupTableData

from vivarium_public_health.risks import Risk
from vivarium_public_health.risks.calibration_constant import (
    get_calibration_constant_pipeline_name,
)
from vivarium_public_health.risks.data_transformations import (
    load_exposure_data,
    pivot_categorical,
)
from vivarium_public_health.risks.distributions import MissingDataError
from vivarium_public_health.utilities import EntityString, TargetString


class RiskEffect(Component):
    """A component to model the effect of a risk factor on an affected entity's target rate.

    This component can source data either from builder.data or from parameters
    supplied in the configuration.

    For a risk named 'risk' that affects  'affected_risk' and 'affected_cause',
    the configuration would look like:

    .. code-block:: yaml

       configuration:
            risk_effect.risk_name_on_affected_target:
               exposure_parameters: 2
               incidence_rate: 10

    """

    ###############
    # Properties #
    ##############

    @property
    def name(self) -> str:
        """The name of this risk effect component."""
        return self.get_name(self.risk, self.target)

    @staticmethod
    def get_name(risk: EntityString, target: TargetString) -> str:
        """Construct the canonical name for a risk effect.

        Parameters
        ----------
        risk
            The entity string identifying the risk factor.
        target
            The target string identifying the affected entity and measure.

        Returns
        -------
            The risk effect name in the form
            ``"risk_effect.{risk_name}_on_{target}"``.
        """
        return f"risk_effect.{risk.name}_on_{target}"

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        """Default configuration values for this component.

        Configuration structure::

            {risk_effect_name}:
                data_sources:
                    relative_risk:
                        Source for relative risk data. Default is the artifact
                        key ``{risk}.relative_risk``. Can also be:
                        - A scalar value (e.g., ``1.5``)
                        - A scipy.stats distribution name (e.g., ``"uniform"``)
                          with parameters in ``data_source_parameters``
                    population_attributable_fraction:
                        Source for PAF data. Default is the artifact key
                        ``{risk}.population_attributable_fraction``. Used to
                        adjust the target rate to account for the portion
                        attributable to this risk.
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
                    "relative_risk": f"{self.risk}.relative_risk",
                    "population_attributable_fraction": f"{self.risk}.population_attributable_fraction",
                },
                "data_source_parameters": {
                    "relative_risk": {},
                },
            }
        }

    @property
    def is_exposure_categorical(self) -> bool:
        """Whether the risk exposure uses a categorical distribution."""
        return self._exposure_distribution_type in [
            "dichotomous",
            "ordered_polytomous",
            "unordered_polytomous",
        ]

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, risk: str, target: str):
        """

        Parameters
        ----------
        risk
            Type and name of risk factor, supplied in the form
            "risk_type.risk_name" where risk_type should be singular (e.g.,
            risk_factor instead of risk_factors).
        target
            Type, name, and target rate of entity to be affected by risk factor,
            supplied in the form "entity_type.entity_name.measure"
            where entity_type should be singular (e.g., cause instead of causes).
        """
        super().__init__()
        self.risk = EntityString(risk)
        self.target = TargetString(target)

        self._exposure_distribution_type = None

        self.exposure_name = f"{self.risk.name}.exposure"
        self.target_name = f"{self.target.name}.{self.target.measure}"
        self.relative_risk_name = f"{self.risk.name}_on_{self.target.name}.relative_risk"

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        """Set up relative risk table, PAF data, and register pipelines.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.
        """
        self.relative_risk_table = self.build_rr_lookup_table(builder)
        self.paf_data = self.get_paf_data(builder)

        self._relative_risk_source = self.get_relative_risk_source(builder)
        self.register_relative_risk_pipeline(builder)

        self.register_target_modifier(builder)
        self.register_paf_modifier(builder)

    #################
    # Setup methods #
    #################

    def build_rr_lookup_table(self, builder: Builder) -> LookupTable:
        """Build the relative risk lookup table.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.

        Returns
        -------
            A lookup table mapping demographic parameters to relative
            risk values.
        """
        self._exposure_distribution_type = self.get_distribution_type(builder)

        rr_data = self.load_relative_risk(builder)
        rr_value_cols = None
        if self.is_exposure_categorical:
            rr_data, rr_value_cols = self.process_categorical_data(builder, rr_data)
        return self.build_lookup_table(
            builder, "relative_risk", data_source=rr_data, value_columns=rr_value_cols
        )

    def get_paf_data(self, builder: Builder) -> LookupTableData:
        """Load population attributable fraction data.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.

        Returns
        -------
            The PAF data for this risk-target pair.
        """
        return self.get_filtered_data(
            builder, self.configuration.data_sources.population_attributable_fraction
        )

    def get_distribution_type(self, builder: Builder) -> str:
        """Get the distribution type for the risk from the configuration."""
        risk_exposure_component = self._get_risk_exposure_class(builder)
        if risk_exposure_component.distribution_type:
            return risk_exposure_component.distribution_type
        return risk_exposure_component.get_distribution_type(builder)

    def load_relative_risk(
        self,
        builder: Builder,
        configuration=None,
    ) -> str | float | pd.DataFrame:
        """Load relative risk data from configuration or artifact.

        Attempt to interpret the data source as a ``scipy.stats``
        distribution name first; if that fails, treat it as an artifact
        key.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.
        configuration
            Optional configuration override.  If ``None``, use
            ``self.configuration``.

        Returns
        -------
            The relative risk data as a scalar, string, or DataFrame.

        Raises
        ------
        ConfigurationError
            If the provided distribution parameters are not valid for
            the specified scipy distribution.
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
        self, builder: "Builder", data_source: str | float | pd.DataFrame
    ) -> float | pd.DataFrame:
        """Load data and filter to the target entity and measure.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.
        data_source
            The data source specification (artifact key, scalar, or
            DataFrame).

        Returns
        -------
            The data filtered to this component's target entity and
            measure.
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
        """Pivot categorical relative risk data to wide format.

        If the data is a scalar, create a two-category DataFrame with
        the scalar as ``"cat1"`` and ``1`` as ``"cat2"``.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.
        rr_data
            The relative risk data (scalar or DataFrame).

        Returns
        -------
            A tuple of the pivoted relative risk data and the list of
            category column names.
        """
        if not isinstance(rr_data, pd.DataFrame):
            cat1 = builder.data.load("population.demographic_dimensions")
            cat1["parameter"] = "cat1"
            cat1["value"] = rr_data
            cat2 = cat1.copy()
            cat2["parameter"] = "cat2"
            cat2["value"] = 1
            rr_data = pd.concat([cat1, cat2], ignore_index=True)
        if "parameter" in rr_data.index.names:
            rr_data = rr_data.reset_index("parameter")

        rr_value_cols = list(rr_data["parameter"].unique())
        rr_data = pivot_categorical(rr_data, "parameter")
        return rr_data, rr_value_cols

    # todo currently this isn't being called. we need to properly set rrs if
    #  the exposure has been rebinned
    def rebin_relative_risk_data(
        self, builder: Builder, relative_risk_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Rebin relative risk data when a polytomous risk is rebinned.

        After rebinning, the relative risk for both exposed and unexposed
        categories is the weighted sum of relative risks of the component
        categories, where weights are relative proportions of exposure
        for those categories.  For example, if cat1, cat2, cat3 are
        exposed categories and cat4 is unexposed with exposure
        [0.1, 0.2, 0.3, 0.4] and matching rr = [rr1, rr2, rr3, 1],
        the rebinned rr for the rebinned cat1 is
        ``(0.1 * rr1 + 0.2 * rr2 + 0.3 * rr3) / (0.1 + 0.2 + 0.3)``.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.
        relative_risk_data
            The relative risk data to potentially rebin.

        Returns
        -------
            The (possibly rebinned) relative risk data.
        """
        if not self.risk in builder.configuration.to_dict():
            return relative_risk_data

        rebin_exposed_categories = set(builder.configuration[self.risk]["rebinned_exposed"])

        if rebin_exposed_categories:
            # todo make sure this works
            exposure_data = load_exposure_data(builder, self.risk)
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
        """Rebin relative risk data using exposure-weighted averaging."""
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
        """Build a callable that computes relative risk for each simulant.

        For continuous exposures, compute relative risk using the TMRED
        and exposure scalar.  For categorical exposures, look up the
        relative risk corresponding to each simulant's category.

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
            tmred = builder.data.load(f"{self.risk}.tmred")
            tmrel = 0.5 * (tmred["min"] + tmred["max"])
            scale = builder.data.load(f"{self.risk}.relative_risk_scalar")

            def generate_relative_risk(index: pd.Index) -> pd.Series:
                """Compute continuous relative risk from TMRED scaling."""
                rr = self.relative_risk_table(index)
                exposure = self.population_view.get_attributes(index, self.exposure_name)
                relative_risk = np.maximum(rr.values ** ((exposure - tmrel) / scale), 1)
                return relative_risk

        else:
            index_columns = ["index", self.risk.name]

            def generate_relative_risk(index: pd.Index) -> pd.Series:
                """Compute categorical relative risk by matching exposure."""
                rr = self.relative_risk_table(index)
                exposure = self.population_view.get_attributes(
                    index, self.exposure_name
                ).reset_index()
                exposure.columns = index_columns
                exposure = exposure.set_index(index_columns)

                relative_risk = rr.stack().reset_index()
                relative_risk.columns = index_columns + ["value"]
                relative_risk = relative_risk.set_index(index_columns)

                effect = relative_risk.loc[exposure.index, "value"].droplevel(self.risk.name)
                return effect

        return generate_relative_risk

    def register_relative_risk_pipeline(self, builder: Builder) -> None:
        """Register the relative risk pipeline.

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

    def register_paf_modifier(self, builder: Builder) -> None:
        """Register the PAF as a modifier on the calibration constant pipeline.

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

    def _get_risk_exposure_class(self, builder: Builder) -> Risk:
        """Look up and validate the Risk component for this risk effect."""
        risk_exposure_component = builder.components.get_component(self.risk)
        if not isinstance(risk_exposure_component, Risk):
            raise ValueError(
                f"Risk effect model {self.name} requires a Risk component named {self.risk}"
            )
        return risk_exposure_component


class NonLogLinearRiskEffect(RiskEffect):
    """A component to model the exposure-parametrized effect of a risk factor.

    More specifically, this models the effect of the risk factor on the target rate of
    some affected entity.

    This component:
    1) reads TMRED data from the artifact and define the TMREL
    2) calculates the relative risk at TMREL by linearly interpolating over
    relative risk data defined in the configuration
    3) divides relative risk data from configuration by RR at TMREL
    and clip to be greater than 1
    4) builds a LookupTable which returns the exposure and RR of the left and right edges
    of the RR bin containing a simulant's exposure
    5) uses this LookupTable to modify the target pipeline by linearly interpolating
    a simulant's RR value and multiplying it by the intended target rate

    """

    ##############
    # Properties #
    ##############

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        """Default configuration values for this component.

        Configuration structure::

            {risk_effect_name}:
                data_sources:
                    relative_risk:
                        Source for relative risk data. Default is the artifact
                        key ``{risk}.relative_risk``. The data must be a
                        DataFrame with a numeric ``parameter`` column containing
                        exposure thresholds and a ``value`` column with the
                        corresponding relative risks.
                    population_attributable_fraction:
                        Source for PAF data. Default is the artifact key
                        ``{risk}.population_attributable_fraction``. Used to
                        adjust the target rate to account for the portion
                        attributable to this risk.
        """
        return {
            self.name: {
                "data_sources": {
                    "relative_risk": f"{self.risk}.relative_risk",
                    "population_attributable_fraction": f"{self.risk}.population_attributable_fraction",
                },
            }
        }

    #################
    # Setup methods #
    #################

    @staticmethod
    def get_name(risk: EntityString, target: TargetString) -> str:
        """Construct the canonical name for a non-log-linear risk effect.

        Parameters
        ----------
        risk
            The entity string identifying the risk factor.
        target
            The target string identifying the affected entity and measure.

        Returns
        -------
            The risk effect name in the form
            ``"non_log_linear_risk_effect.{risk_name}_on_{target}"``.
        """
        return f"non_log_linear_risk_effect.{risk.name}_on_{target}"

    def build_rr_lookup_table(self, builder: Builder) -> LookupTable:
        """Build a lookup table mapping exposure intervals to relative risks.

        Define left and right edges of exposure bins and their
        corresponding relative risk values for piecewise linear
        interpolation.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.

        Returns
        -------
            A lookup table with columns for left/right exposure and
            left/right relative risk values.
        """
        rr_data = self.load_relative_risk(builder)
        self.validate_rr_data(rr_data)

        def define_rr_intervals(df: pd.DataFrame) -> pd.DataFrame:
            """Create left/right exposure and RR interval columns."""
            # create new row for right-most exposure bin (RR is same as max RR)
            max_exposure_row = df.tail(1).copy()
            max_exposure_row["parameter"] = np.inf
            rr_data = pd.concat([df, max_exposure_row]).reset_index()

            rr_data["left_exposure"] = [0] + rr_data["parameter"][:-1].tolist()
            rr_data["left_rr"] = [rr_data["value"].min()] + rr_data["value"][:-1].tolist()
            rr_data["right_exposure"] = rr_data["parameter"]
            rr_data["right_rr"] = rr_data["value"]

            return rr_data[
                ["parameter", "left_exposure", "left_rr", "right_exposure", "right_rr"]
            ]

        # define exposure and rr interval columns
        demographic_cols = [
            col for col in rr_data.columns if col != "parameter" and col != "value"
        ]
        rr_data = (
            rr_data.groupby(demographic_cols)
            .apply(define_rr_intervals)
            .reset_index(level=-1, drop=True)
            .reset_index()
        )
        rr_data = rr_data.drop("parameter", axis=1)
        rr_data[f"{self.risk.name}_exposure_for_non_loglinear_riskeffect_start"] = rr_data[
            "left_exposure"
        ]
        rr_data[f"{self.risk.name}_exposure_for_non_loglinear_riskeffect_end"] = rr_data[
            "right_exposure"
        ]
        # build lookup table
        rr_value_cols = ["left_exposure", "left_rr", "right_exposure", "right_rr"]
        return self.build_lookup_table(
            builder, "relative_risk", data_source=rr_data, value_columns=rr_value_cols
        )

    def load_relative_risk(
        self,
        builder: Builder,
        configuration: LayeredConfigTree | None = None,
    ) -> str | float | pd.DataFrame:
        """Load relative risk data, normalizing by RR at the TMREL.

        Compute the Theoretical Minimum-Risk Exposure Level (TMREL)
        from TMRED data, interpolate RR at the TMREL, divide all RR
        values by this quantity, and clip to be at least 1.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.
        configuration
            Optional configuration override.  If ``None``, use
            ``self.configuration``.

        Returns
        -------
            The normalized relative risk data as a DataFrame.

        Raises
        ------
        MissingDataError
            If the TMRED data uses draw-level TMRELs or is not found.
        """
        if configuration is None:
            configuration = self.configuration

        # get TMREL
        tmred = builder.data.load(f"{self.risk}.tmred")
        if tmred["distribution"] == "uniform":
            draw = builder.configuration.input_data.input_draw_number
            rng = np.random.default_rng(builder.randomness.get_seed(self.name + str(draw)))
            self.tmrel = rng.uniform(tmred["min"], tmred["max"])
        elif tmred["distribution"] == "draws":  # currently only for iron deficiency
            raise MissingDataError(
                f"This data has draw-level TMRELs. You will need to contact the research team that models {self.risk.name} to get this data."
            )
        else:
            raise MissingDataError(f"No TMRED found in gbd_mapping for risk {self.risk.name}")

        # calculate RR at TMREL
        rr_source = configuration.data_sources.relative_risk
        original_rrs = self.get_filtered_data(builder, rr_source)

        self.validate_rr_data(original_rrs)

        demographic_cols = [
            col for col in original_rrs.columns if col != "parameter" and col != "value"
        ]

        def get_rr_at_tmrel(rr_data: pd.DataFrame) -> float:
            """Interpolate the relative risk at the TMREL."""
            interpolated_rr_function = scipy.interpolate.interp1d(
                rr_data["parameter"],
                rr_data["value"],
                kind="linear",
                bounds_error=False,
                fill_value=(
                    rr_data["value"].min(),
                    rr_data["value"].max(),
                ),
            )
            rr_at_tmrel = interpolated_rr_function(self.tmrel).item()
            return rr_at_tmrel

        rrs_at_tmrel = (
            original_rrs.groupby(demographic_cols)
            .apply(get_rr_at_tmrel)
            .rename("rr_at_tmrel")
        )
        rr_data = original_rrs.merge(rrs_at_tmrel.reset_index())
        rr_data["value"] = rr_data["value"] / rr_data["rr_at_tmrel"]
        rr_data["value"] = np.clip(rr_data["value"], 1.0, np.inf)
        rr_data = rr_data.drop("rr_at_tmrel", axis=1)

        return rr_data

    def get_relative_risk_source(self, builder: Builder) -> Callable[[pd.Index], pd.Series]:
        """Build a callable that interpolates relative risk from exposure.

        Use piecewise linear interpolation within the exposure bins
        defined by the relative risk lookup table.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.

        Returns
        -------
            A callable that accepts a simulant index and returns
            interpolated relative risk values.
        """

        def generate_relative_risk(index: pd.Index) -> pd.Series:
            """Interpolate relative risk from exposure within RR bins."""
            rr_intervals = self.relative_risk_table(index)
            # NOTE: We are calling the cached exposure pipeline here for performance
            # purposes (as opposed to the f{self.risk.name}.exposure pipeline itself).
            exposure = self.population_view.get_attributes(
                index, f"{self.risk.name}_exposure_for_non_loglinear_riskeffect"
            )
            x1, x2 = (
                rr_intervals["left_exposure"].values,
                rr_intervals["right_exposure"].values,
            )
            y1, y2 = rr_intervals["left_rr"].values, rr_intervals["right_rr"].values
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            relative_risk = b + m * exposure
            return relative_risk

        return generate_relative_risk

    ##############
    # Validators #
    ##############

    def validate_rr_data(self, rr_data: pd.DataFrame) -> None:
        """Validate the relative risk data for non-log-linear effects.

        Verify that the ``parameter`` column contains numeric data and
        that values are monotonically increasing within each demographic
        group.

        Parameters
        ----------
        rr_data
            The relative risk data to validate.

        Raises
        ------
        ValueError
            If the ``parameter`` column is not numeric or is not
            monotonically increasing within demographic groups.
        """
        # check that rr_data has numeric parameter data
        parameter_data_is_numeric = rr_data["parameter"].dtype.kind in "biufc"
        if not parameter_data_is_numeric:
            raise ValueError(
                f"The parameter column in your {self.risk.name} relative risk data must contain numeric data. Its dtype is {rr_data['parameter'].dtype} instead."
            )

        # and that these RR values are monotonically increasing within each demographic group
        # so that each simulant's exposure will assign them to either one bin or one RR value
        demographic_cols = [
            col for col in rr_data.columns if col != "parameter" and col != "value"
        ]

        def values_are_monotonically_increasing(df: pd.DataFrame) -> bool:
            """Check if parameter values are monotonically increasing."""
            return np.all(df["parameter"].values[1:] >= df["parameter"].values[:-1])

        group_is_increasing = rr_data.groupby(demographic_cols).apply(
            values_are_monotonically_increasing, include_groups=False
        )
        if not group_is_increasing.all():
            raise ValueError(
                "The parameter column in your relative risk data must be monotonically increasing to be used in NonLogLinearRiskEffect."
            )
