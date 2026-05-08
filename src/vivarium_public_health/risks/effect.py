"""
==================
Risk Effect Models
==================

This module contains tools for modeling the relationship between risk
exposure models and disease models.

"""

from collections.abc import Callable
from functools import partial
from typing import Any, Literal

import numpy as np
import pandas as pd
import scipy
from layered_config_tree import LayeredConfigTree
from vivarium.framework.engine import Builder
from vivarium.framework.lookup import LookupTable
from vivarium.types import DataInput

from vivarium_public_health.causal_factor.distributions import MissingDataError
from vivarium_public_health.causal_factor.effect import CausalFactorEffect
from vivarium_public_health.risks import Risk
from vivarium_public_health.utilities import EntityString, TargetString


class RiskEffect(CausalFactorEffect):
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

    EXPOSURE_CLASS = Risk

    ##############
    # Properties #
    ##############

    @staticmethod
    def get_name(risk: EntityString, target: TargetString) -> str:
        """The name of this risk effect component."""
        return f"risk_effect.{risk.name}_on_{target}"

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        """Default configuration values for this component.

        The ``effect`` and ``calibration_constant`` keys are the "real" keys
        consumed by the underlying component. By default they redirect to the
        ``relative_risk`` and ``population_attributable_fraction`` keys, which
        in turn default to the corresponding artifact keys. Users can override
        either layer: configure ``relative_risk`` / ``population_attributable_fraction``
        to keep the redirect, or configure ``effect`` / ``calibration_constant``
        directly to bypass it.

        Configuration structure::

            {risk_effect_name}:
                data_sources:
                    effect:
                        Source for effect data. Defaults to the
                        ``relative_risk`` key below. Override directly to
                        bypass the relative-risk redirect.
                    calibration_constant:
                        Source for calibration constant data. Defaults to the
                        ``population_attributable_fraction`` key below.
                        Override directly to bypass the PAF redirect.
                    relative_risk:
                        Source for relative risk data. Default is the artifact
                        key ``{risk}.relative_risk``. Can also be:
                        - A scalar value (e.g., ``1.5``)
                        - A scipy.stats distribution name (e.g., ``"uniform"``)
                          with parameters in ``data_source_parameters``
                    population_attributable_fraction:
                        Source for PAF data. Default is the artifact key
                        ``{risk}.population_attributable_fraction``. Used to
                        adjust the target measure to account for the portion
                        attributable to this risk.
                data_source_parameters:
                    effect: dict
                        Defaults to the ``relative_risk`` parameters below.
                    relative_risk: dict
                        Parameters for scipy.stats distributions when using
                        a distribution name as the ``relative_risk`` source.
                        For example, ``{"loc": 1.0, "scale": 0.5}`` for a
                        uniform distribution.
        """

        def _redirect_source(builder: Builder, key: str) -> DataInput:
            """Redirect data source to the appropriate configuration key."""
            return self.get_data(builder, self.configuration.data_sources[key])

        return {
            self.name: {
                "data_sources": {
                    "effect": partial(_redirect_source, key="relative_risk"),
                    "calibration_constant": partial(
                        _redirect_source,
                        key="population_attributable_fraction",
                    ),
                    "relative_risk": f"{self.risk}.relative_risk",
                    "population_attributable_fraction": f"{self.risk}.population_attributable_fraction",
                },
                "data_source_parameters": {
                    "effect": {},
                    "relative_risk": {},
                },
            }
        }

    @property
    def risk(self) -> str:
        """The type and name of a risk, specified as "type.name". Type is singular."""
        return self.causal_factor

    @property
    def effect_parameters(self) -> dict[str, Any]:
        """Effect parameters for this component."""
        params = (
            self.configuration.data_source_parameters.effect
            or self.configuration.data_source_parameters.relative_risk
        )
        return params.to_dict()

    #####################
    # Lifecycle methods #
    #####################

    def __init__(
        self,
        risk: str,
        target: str,
        effect_type: Literal["multiplicative", "additive"] = "multiplicative",
    ) -> None:
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
        effect_type
            The type of effect model to use, either "multiplicative" or "additive".
            This determines how the effect data modifies the target rate. Default
            is "multiplicative".
        """
        super().__init__(risk, target, effect_type)


class NonLogLinearRiskEffect(RiskEffect):
    """A component to model the exposure-parametrized effect of a risk factor.

    More specifically, this models the effect of the risk factor on the target rate of
    some affected entity.

    This component:

    1. Reads TMRED data from the artifact and defines the TMREL.
    2. Calculates the relative risk at TMREL by linearly interpolating over
       relative risk data defined in the configuration.
    3. Divides relative risk data from configuration by RR at TMREL
       and clips to be greater than 1.
    4. Builds a ``LookupTable`` that returns the exposure and RR of the left
       and right edges of the RR bin containing a simulant's exposure.
    5. Uses this ``LookupTable`` to modify the target pipeline by linearly
       interpolating a simulant's RR value and multiplying it by the intended
       target rate.

    """

    #################
    # Setup methods #
    #################

    @property
    def name(self) -> str:
        """The name of this non-log-linear risk effect component."""
        return f"non_log_linear_risk_effect.{self.causal_factor.name}_on_{self.target}"

    def build_effect_lookup_table(self, builder: Builder) -> LookupTable:
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
        rr_data = self.load_effect_data(builder)
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
        rr_data[
            f"{self.causal_factor.name}_exposure_for_non_loglinear_riskeffect_start"
        ] = rr_data["left_exposure"]
        rr_data[
            f"{self.causal_factor.name}_exposure_for_non_loglinear_riskeffect_end"
        ] = rr_data["right_exposure"]
        # build lookup table
        rr_value_cols = ["left_exposure", "left_rr", "right_exposure", "right_rr"]
        return self.build_lookup_table(
            builder, "relative_risk", data_source=rr_data, value_columns=rr_value_cols
        )

    def load_effect_data(
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
        tmred = builder.data.load(f"{self.causal_factor}.tmred")
        if tmred["distribution"] == "uniform":
            draw = builder.configuration.input_data.input_draw_number
            rng = np.random.default_rng(builder.randomness.get_seed(self.name + str(draw)))
            self.tmrel = rng.uniform(tmred["min"], tmred["max"])
        elif tmred["distribution"] == "draws":  # currently only for iron deficiency
            raise MissingDataError(
                f"This data has draw-level TMRELs. You will need to contact the research team that models {self.causal_factor.name} to get this data."
            )
        else:
            raise MissingDataError(
                f"No TMRED found in gbd_mapping for risk {self.causal_factor.name}"
            )

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

    def get_effect_source(self, builder: Builder) -> Callable[[pd.Index], pd.Series]:
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
            rr_intervals = self.effect_table(index)
            # NOTE: We are calling the cached exposure pipeline here for performance
            # purposes (as opposed to the f{self.causal_factor.name}.exposure pipeline itself).
            exposure = self.population_view.get(
                index, f"{self.causal_factor.name}_exposure_for_non_loglinear_riskeffect"
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
                f"The parameter column in your {self.causal_factor.name} relative risk data must contain numeric data. Its dtype is {rr_data['parameter'].dtype} instead."
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
