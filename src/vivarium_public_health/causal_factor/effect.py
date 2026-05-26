"""
==========================
CausalFactor Effect Models
==========================

This module contains tools for modeling the relationship between causal factor
exposure models and the models they affect.

"""

import functools
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from importlib import import_module
from typing import Any, Literal

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


def _deprecated_method_shim(old_name: str):
    """Forward to a deprecated method on the same class if a subclass still defines it.

    Wrap a renamed method so that subclasses defining the old name continue to work
    with a ``DeprecationWarning``. Once the old name is removed, drop the decorator.
    """

    def decorator(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            if hasattr(self, old_name):
                warnings.warn(
                    f"{self.__class__.__name__} defines `{old_name}`, which is deprecated. "
                    f"Rename to `{method.__name__}`.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                return getattr(self, old_name)(*args, **kwargs)
            return method(self, *args, **kwargs)

        return wrapper

    return decorator


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
    @abstractmethod
    def get_name(causal_factor: EntityString, target: TargetString) -> str:
        """Return the component name for a causal factor and target pair."""
        ...

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        """Default configuration values for this component.

        Configuration structure::

            {causal_factor_effect_name}:
                data_sources:
                    effect:
                        Source for effect data. Default is the artifact
                        key ``{causal_factor}.effect``. Can also be:
                        - A scalar value (e.g., ``1.5``)
                        - A scipy.stats distribution name (e.g., ``"uniform"``)
                          with parameters in ``data_source_parameters``
                    calibration_constant:
                        Source for calibration constant data. Default is the
                        artifact key ``{causal_factor}.calibration_constant``.
                        Used to adjust the target measure to account for the
                        portion attributable to this causal factor.
                data_source_parameters:
                    effect: dict
                        Parameters for scipy.stats distributions when using
                        a distribution name as the ``effect`` source.
                        For example, ``{"loc": 1.0, "scale": 0.5}`` for a
                        uniform distribution.
        """
        return {
            self.name: {
                "data_sources": {
                    "effect": f"{self.causal_factor}.effect",
                    "calibration_constant": f"{self.causal_factor}.calibration_constant",
                },
                "data_source_parameters": {
                    "effect": {},
                },
            }
        }

    @property
    def effect_parameters(self) -> dict[str, Any]:
        """Effect parameters for this component."""
        return self.configuration.data_source_parameters.effect.to_dict()

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

    def __init__(
        self,
        causal_factor: str,
        target: str,
        effect_type: Literal["multiplicative", "additive"] = "multiplicative",
    ):
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
        effect_type
            The type of effect model to use, either "multiplicative" or "additive".
            This determines how the effect data modifies the target measure.
        """
        super().__init__()
        self.causal_factor = EntityString(causal_factor)
        self.target = TargetString(target)
        self.effect_type = effect_type

        self._exposure_distribution_type = None

        self.exposure_name = f"{self.causal_factor.name}.exposure"
        self.target_name = f"{self.target.name}.{self.target.measure}"
        effect_type_name = (
            "relative_risk" if self.effect_type == "multiplicative" else "additive_effect"
        )
        self.effect_name = (
            f"{self.causal_factor.name}_on_{self.target_name}.{effect_type_name}"
        )

    def setup(self, builder: Builder) -> None:
        """Set up the causal factor effect component.

        Load distribution type and PAF data, define effect source,
        build effect lookup tables, register effect pipeline,
        and register target and calibration constant modifiers.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.
        """
        self._exposure_distribution_type = self.get_distribution_type(builder)
        self.effect_table = self.build_effect_lookup_table(builder)
        self.calibration_constant_data = self.load_calibration_constant_data(builder)

        self._effect_source = self.get_effect_source(builder)
        self.register_effect_pipeline(builder)

        self.register_target_modifier(builder)
        self.register_calibration_constant_modifier(builder)

    #################
    # Setup methods #
    #################

    @_deprecated_method_shim("build_rr_lookup_table")
    def build_effect_lookup_table(self, builder: Builder) -> LookupTable:
        """Build a lookup table for effect data.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.

        Returns
        -------
            A lookup table of effect values.
        """
        effect_data = self.load_effect_data(builder)
        if self.is_exposure_categorical:
            effect_data = self.process_categorical_data(builder, effect_data)

        return self.build_lookup_table(
            builder, self.effect_type, data_source=effect_data
        )

    def load_calibration_constant_data(self, builder: Builder) -> LookupTableData:
        """Load calibration constant data for this effect.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.

        Returns
        -------
            The calibration constant data.
        """
        return self.get_filtered_data(
            builder, self.configuration.data_sources.calibration_constant
        )

    def get_distribution_type(self, builder: Builder) -> str:
        """Get the distribution type for the causal factor from the configuration."""
        causal_factor_exposure_component = self._get_causal_factor_exposure_component(builder)
        return (
            causal_factor_exposure_component.distribution_type
            or causal_factor_exposure_component.get_distribution_type(builder)
        )

    @_deprecated_method_shim("load_relative_risk")
    def load_effect_data(
        self,
        builder: Builder,
        configuration=None,
    ) -> str | float | pd.DataFrame:
        """Load effect data from the configuration.

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
            The effect data.

        Raises
        ------
        ConfigurationError
            If the distribution parameters are invalid.
        """
        if configuration is None:
            configuration = self.configuration

        effect_source = configuration.data_sources.effect

        if isinstance(effect_source, str):
            try:
                distribution = getattr(import_module("scipy.stats"), effect_source)
                rng = np.random.default_rng(builder.randomness.get_seed(self.name))
                effect_data = distribution(**self.effect_parameters).ppf(rng.random())
            except AttributeError:
                effect_data = self.get_filtered_data(builder, effect_source)
            except TypeError:
                raise ConfigurationError(
                    f"Parameters {self.effect_parameters} are not valid for distribution {effect_source}."
                )
        else:
            effect_data = self.get_filtered_data(builder, effect_source)
        return effect_data

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
            correct_target_mask = pd.Series(True, index=data.index)
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
        self, builder: Builder, effect_data: float | pd.DataFrame
    ) -> pd.DataFrame:
        """Process effect data for categorical exposures.

        For scalar effect data with a dichotomous distribution, construct a
        DataFrame with exposed/unexposed categories. Return a DataFrame with
        lookup dimensions on the row index and exposure categories as columns.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.
        effect_data
            The effect data.

        Returns
        -------
            The processed effect data, with lookup dimensions on the row
            index and exposure categories as columns.

        Raises
        ------
        ValueError
            If scalar effect data is provided with a non-dichotomous
            distribution.
        """
        if not isinstance(effect_data, pd.DataFrame):
            if self._exposure_distribution_type != "dichotomous":
                raise ValueError(
                    f"Effect data for categorical exposure must be a DataFrame unless the "
                    f"exposure distribution is dichotomous. Found type {type(effect_data)} with "
                    f"exposure distribution type {self._exposure_distribution_type}."
                )
            causal_factor_type = self.causal_factor.type
            exposed = DichotomousDistribution.get_exposed(causal_factor_type)
            unexposed = DichotomousDistribution.get_unexposed(causal_factor_type)
            demographic_dimensions = builder.data.load("population.demographic_dimensions")

            effect_data_ = pd.DataFrame(
                index=pd.MultiIndex.from_frame(demographic_dimensions)
            )
            effect_data_[exposed] = effect_data
            effect_data_[unexposed] = 1
            return effect_data_

        if "parameter" in effect_data.index.names:
            effect_data = effect_data.reset_index("parameter")
        index_cols = [c for c in effect_data.columns if c not in ("parameter", "value")]
        effect_data = effect_data.pivot(
            index=index_cols, columns="parameter", values="value"
        )
        effect_data.columns.name = None

        if self._exposure_distribution_type == "dichotomous":
            effect_data = DichotomousDistribution.rename_deprecated_categories(
                self.causal_factor.type, effect_data
            )
        return effect_data

    # todo currently this isn't being called. we need to properly set rrs if
    #  the exposure has been rebinned
    @_deprecated_method_shim("rebin_relative_risk_data")
    def rebin_effect_data(self, builder, effect_data: pd.DataFrame) -> pd.DataFrame:
        """Rebin effect data.

        When the polytomous effect is rebinned, matching effect data needs to be rebinned.
        After rebinning, effect data for both exposed and unexposed categories should be the weighted sum of effect data
        of the component categories where weights are relative proportions of exposure of those categories.
        For example, if cat1, cat2, cat3 are exposed categories and cat4 is unexposed with exposure [0.1,0.2,0.3,0.4],
        for the matching effect data = [effect1, effect2, effect3, 1], rebinned effect data for the rebinned cat1 should be:
        (0.1 *effect1 + 0.2 * effect2 + 0.3* effect3) / (0.1+0.2+0.3)
        """
        if not self.causal_factor in builder.configuration.to_dict():
            return effect_data

        rebin_exposed_categories = set(
            builder.configuration[self.causal_factor]["rebinned_exposed"]
        )

        if rebin_exposed_categories:
            # todo make sure this works
            exposure_data = load_exposure_data(builder, self.causal_factor)
            effect_data = self._rebin_effect_data(
                effect_data, exposure_data, rebin_exposed_categories
            )

        return effect_data

    @_deprecated_method_shim("_rebin_relative_risk_data")
    def _rebin_effect_data(
        self,
        effect_data: pd.DataFrame,
        exposure_data: pd.DataFrame,
        rebin_exposed_categories: set,
    ) -> pd.DataFrame:
        """Compute exposure-weighted relative risks for rebinned categories."""
        cols = list(exposure_data.columns.difference(["value"]))

        effect_data = effect_data.merge(exposure_data, on=cols)
        effect_data["value_x"] = effect_data.value_x.multiply(effect_data.value_y)
        effect_data.parameter = effect_data["parameter"].map(
            lambda p: "cat1" if p in rebin_exposed_categories else "cat2"
        )
        effect_data = effect_data.groupby(cols).sum().reset_index()
        effect_data["value"] = effect_data.value_x.divide(effect_data.value_y).fillna(0)
        return effect_data.drop(columns=["value_x", "value_y"])

    @_deprecated_method_shim("get_relative_risk_source")
    def get_effect_source(self, builder: Builder) -> Callable[[pd.Index], pd.Series]:
        """Build a callable that computes effect data from exposure.

        For continuous exposures, use TMRED-based log-linear scaling.
        For categorical exposures, look up the effect data for each simulant's
        exposure category.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.

        Returns
        -------
            A callable that accepts a simulant index and returns
            effect data values.
        """
        if not self.is_exposure_categorical:
            tmred = builder.data.load(f"{self.causal_factor}.tmred")
            tmrel = 0.5 * (tmred["min"] + tmred["max"])
            scale = builder.data.load(f"{self.causal_factor}.relative_risk_scalar")

            def generate_effect(index: pd.Index) -> pd.Series:
                effect = self.effect_table(index)
                exposure = self.population_view.get(index, self.exposure_name)
                effect = np.maximum(effect.values ** ((exposure - tmrel) / scale), 1)
                return effect

        else:
            index_columns = ["index", self.causal_factor.name]

            def generate_effect(index: pd.Index) -> pd.Series:
                effect = self.effect_table(index)
                exposure = self.population_view.get(index, self.exposure_name).reset_index()
                exposure.columns = index_columns
                exposure = exposure.set_index(index_columns)

                effect = effect.stack().reset_index()
                effect.columns = index_columns + ["value"]
                effect = effect.set_index(index_columns)

                effect = effect.loc[exposure.index, "value"].droplevel(
                    self.causal_factor.name
                )
                return effect

        return generate_effect

    @_deprecated_method_shim("register_relative_risk_pipeline")
    def register_effect_pipeline(self, builder: Builder) -> None:
        """Register the effect pipeline with the simulation.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.
        """
        builder.value.register_attribute_producer(
            self.effect_name,
            self._effect_source,
            required_resources=[self.exposure_name],
        )

    def register_target_modifier(self, builder: Builder) -> None:
        """Register the effect as a modifier on the target pipeline.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.
        """
        builder.value.register_attribute_modifier(self.target_name, modifier=self.effect_name)

    def register_calibration_constant_modifier(self, builder: Builder) -> None:
        """Register the calibration constant data as a modifier on the calibration constant pipeline.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.
        """
        builder.value.register_value_modifier(
            get_calibration_constant_pipeline_name(self.target_name),
            modifier=lambda: self.calibration_constant_data,
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
                f"{self.__class__.__name__} model {self.name} requires a {self.EXPOSURE_CLASS.__name__} "
                f"component named '{self.causal_factor}'."
            )
        return causal_factor_exposure_component
