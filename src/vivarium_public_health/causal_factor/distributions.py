"""
=================================
Risk Exposure Distribution Models
=================================

This module contains tools for modeling several different risk
exposure distributions.

"""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import NamedTuple

import numpy as np
import pandas as pd
import risk_distributions as rd
from layered_config_tree import LayeredConfigTree
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.lookup import DEFAULT_VALUE_COLUMN, LookupTable
from vivarium.framework.population import SimulantData

from vivarium_public_health.causal_factor.calibration_constant import (
    register_risk_affected_attribute_producer,
)
from vivarium_public_health.causal_factor.utilities import pivot_categorical
from vivarium_public_health.utilities import EntityString


class MissingDataError(Exception):
    """Custom exception for missing data."""

    pass


class CausalFactorDistribution(Component, ABC):
    """Abstract base class for causal factor exposure distribution models.

    Subclasses implement specific distribution types (e.g., continuous,
    polytomous, dichotomous, ensemble) for modeling causal factor exposures
    in a simulation.
    """

    #####################
    # Lifecycle methods #
    #####################

    def __init__(
        self,
        causal_factor: EntityString,
        distribution_type: str,
        exposure_data: int | float | pd.DataFrame | None = None,
    ) -> None:
        """
        Parameters
        ----------
        causal_factor
            The entity string identifying the risk factor.
        distribution_type
            The type of distribution (e.g., ``"normal"``,
            ``"dichotomous"``).
        exposure_data
            Optional pre-loaded exposure data.  If ``None``, data is
            loaded from the simulation during setup.
        """
        super().__init__()
        self.causal_factor = causal_factor
        self.distribution_type = distribution_type
        self._exposure_data = exposure_data

        self.causal_factor_propensity = f"{self.causal_factor.name}.propensity"
        self.exposure_ppf_pipeline = f"{self.causal_factor.name}.exposure_distribution.ppf"

    #################
    # Setup methods #
    #################

    def get_configuration(self, builder: "Builder") -> LayeredConfigTree | None:
        """Return the configuration tree for this causal factor.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.

        Returns
        -------
            The configuration sub-tree for this causal factor.
        """
        return builder.configuration[self.causal_factor]

    def get_exposure_data(self, builder: Builder) -> int | float | pd.DataFrame:
        """Return exposure data (using pre-loaded data if available).

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.

        Returns
        -------
            The exposure data for this risk factor.
        """
        if self._exposure_data is not None:
            return self._exposure_data
        return self.get_data(builder, self.configuration["data_sources"]["exposure"])

    def setup(self, builder: Builder) -> None:
        """Register the exposure PPF pipeline.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.
        """
        self.register_exposure_ppf_pipeline(builder)

    @abstractmethod
    def register_exposure_ppf_pipeline(self, builder: Builder) -> None:
        """Register the exposure PPF pipeline with the simulation.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.
        """
        pass


class EnsembleDistribution(CausalFactorDistribution):
    """Model risk exposure using an ensemble of weighted distributions.

    Combine multiple parametric distributions (e.g., normal, log-normal,
    gamma) weighted by GBD-derived weights to represent complex exposure
    distributions.
    """

    #####################
    # Lifecycle methods #
    #####################

    def __init__(
        self, causal_factor: EntityString, distribution_type: str = "ensemble"
    ) -> None:
        """
        Parameters
        ----------
        causal_factor
            The entity string identifying the causal factor.
        distribution_type
            The distribution type label. Default is ``"ensemble"``.
        """
        super().__init__(causal_factor, distribution_type)
        self.ensemble_propensity = f"ensemble_propensity.{self.causal_factor}"

    #################
    # Setup methods #
    #################

    def setup(self, builder: Builder) -> None:
        """Build distribution weight and parameter lookup tables.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.
        """
        distributions, weights, parameters = self.get_distribution_definitions(builder)
        self.distribution_weights_table = self.build_lookup_table(
            builder,
            "exposure_distribution_weights",
            data_source=weights,
            value_columns=distributions,
        )

        self.parameters = {
            parameter: self.build_lookup_table(
                builder,
                parameter,
                data_source=data.reset_index(),
                value_columns=list(data.columns),
            )
            for parameter, data in parameters.items()
        }

        super().setup(builder)
        self.randomness = builder.randomness.get_stream(self.ensemble_propensity)
        builder.population.register_initializer(
            initializer=self.initialize_ensemble_propensity,
            columns=self.ensemble_propensity,
            required_resources=[self.randomness],
        )

    def get_distribution_definitions(
        self, builder: Builder
    ) -> tuple[list[str], pd.DataFrame, dict[str, pd.DataFrame]]:
        """Load and compute ensemble distribution definitions.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.

        Returns
        -------
            A tuple of (distribution names, weights DataFrame,
            parameter dict keyed by distribution name).

        Raises
        ------
        NotImplementedError
            If the ``glnorm`` distribution has non-zero weights.
        """
        exposure_data = self.get_exposure_data(builder)
        standard_deviation = self.get_data(
            builder,
            self.configuration["data_sources"]["exposure_standard_deviation"],
        )
        weights_source = self.configuration["data_sources"]["ensemble_distribution_weights"]
        raw_weights = self.get_data(builder, weights_source)

        glnorm_mask = raw_weights["parameter"] == "glnorm"
        if np.any(raw_weights.loc[glnorm_mask, DEFAULT_VALUE_COLUMN]):
            raise NotImplementedError("glnorm distribution is not supported")
        raw_weights = raw_weights[~glnorm_mask]

        distributions = list(raw_weights["parameter"].unique())

        raw_weights = pivot_categorical(
            raw_weights, pivot_column="parameter", reset_index=False
        )

        weights, parameters = rd.EnsembleDistribution.get_parameters(
            raw_weights,
            mean=get_risk_distribution_parameter(exposure_data),
            sd=get_risk_distribution_parameter(standard_deviation),
        )
        return distributions, weights.reset_index(), parameters

    def register_exposure_ppf_pipeline(self, builder: Builder) -> None:
        """Register the ensemble exposure PPF pipeline.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.
        """
        tables = [self.distribution_weights_table, *self.parameters.values()]
        register_risk_affected_attribute_producer(
            builder=builder,
            name=self.exposure_ppf_pipeline,
            source=self.exposure_ppf,
            required_resources=[
                *tables,
                self.causal_factor_propensity,
                self.ensemble_propensity,
            ],
        )

    ########################
    # Event-driven methods #
    ########################

    def initialize_ensemble_propensity(self, pop_data: SimulantData) -> None:
        """Initialize propensities for selecting child distributions in the ensemble.

        Parameters
        ----------
        pop_data
            Metadata about the simulants being initialized.
        """
        ensemble_propensity = self.randomness.get_draw(pop_data.index).rename(
            self.ensemble_propensity
        )
        self.population_view.initialize(ensemble_propensity)

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def exposure_ppf(self, index: pd.Index) -> pd.Series:
        """Calculate exposure values from propensities using the ensemble.

        Parameters
        ----------
        index
            An index representing the simulants.

        Returns
        -------
            A series of exposure values.
        """
        pop = self.population_view.get(
            index, [self.causal_factor_propensity, self.ensemble_propensity]
        )
        quantiles = pop[self.causal_factor_propensity]

        if not pop.empty:
            quantiles = clip(quantiles)
            weights = self.distribution_weights_table(quantiles.index)
            parameters = {
                name: param(quantiles.index) for name, param in self.parameters.items()
            }
            x = rd.EnsembleDistribution(weights, parameters).ppf(
                quantiles, pop[self.ensemble_propensity]
            )
            x[x.isnull()] = 0
        else:
            x = pd.Series([])
        return x


class ContinuousDistribution(CausalFactorDistribution):
    """Model risk exposure using a continuous parametric distribution.

    Support ``"normal"`` and ``"lognormal"`` distribution types.  Exposure
    values are derived from the distribution's percent-point function
    evaluated at each simulant's propensity.
    """

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, causal_factor: EntityString, distribution_type: str) -> None:
        """
        Parameters
        ----------
        causal_factor
            The entity string identifying the causal factor.
        distribution_type
            The distribution type (``"normal"`` or ``"lognormal"``).

        Raises
        ------
        NotImplementedError
            If the distribution type is not ``"normal"`` or
            ``"lognormal"``.
        """
        super().__init__(causal_factor, distribution_type)
        self.exposure_params_name = f"{self.causal_factor}.exposure_parameters"
        self.standard_deviation = None
        try:
            self._distribution = {
                "normal": rd.Normal,
                "lognormal": rd.LogNormal,
            }[distribution_type]
        except KeyError:
            raise NotImplementedError(
                f"Distribution type {distribution_type} is not supported for "
                f"causal_factor {self.causal_factor.name}."
            )

    #################
    # Setup methods #
    #################

    def setup(self, builder):
        """Compute distribution parameters and register pipelines.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.
        """
        parameters = self.get_distribution_parameters(builder)
        self.parameters_table = self.build_lookup_table(
            builder,
            "exposure_parameters",
            data_source=parameters.reset_index(),
            value_columns=list(parameters.columns),
        )
        self.register_exposure_params_pipeline(builder)
        super().setup(builder)

    def get_distribution_parameters(self, builder: "Builder") -> None:
        """Compute the distribution parameters from exposure data.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.

        Returns
        -------
            A DataFrame of distribution parameters (e.g., loc, scale).
        """
        exposure_data = self.get_exposure_data(builder)
        standard_deviation = self.get_data(
            builder, self.configuration["data_sources"]["exposure_standard_deviation"]
        )
        return self._distribution.get_parameters(
            mean=get_risk_distribution_parameter(exposure_data),
            sd=get_risk_distribution_parameter(standard_deviation),
        )

    def register_exposure_ppf_pipeline(self, builder: Builder) -> None:
        """Register the continuous exposure PPF pipeline.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.
        """
        register_risk_affected_attribute_producer(
            builder=builder,
            name=self.exposure_ppf_pipeline,
            source=self.exposure_ppf,
            required_resources=[self.exposure_params_name, self.causal_factor_propensity],
        )

    def register_exposure_params_pipeline(self, builder: Builder) -> None:
        """Register the exposure parameters pipeline.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.
        """
        builder.value.register_attribute_producer(
            self.exposure_params_name, source=self.parameters_table
        )

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def exposure_ppf(self, index: pd.Index) -> pd.Series:
        """Calculate exposure values from propensities.

        Parameters
        ----------
        index
            An index representing the simulants.

        Returns
        -------
            A series of exposure values.
        """
        pop = self.population_view.get(
            index, [self.causal_factor_propensity, self.exposure_params_name]
        )
        quantiles = pop[self.causal_factor_propensity]

        if not quantiles.empty:
            quantiles = clip(quantiles)
            x = self._distribution(parameters=pop[self.exposure_params_name]).ppf(quantiles)
            x[x.isnull()] = 0
        else:
            x = pd.Series([])
        return x


class PolytomousDistribution(CausalFactorDistribution):
    """Model risk exposure as a set of ordered or unordered categories.

    Assign each simulant to a category by comparing their propensity
    against the cumulative sum of category exposure probabilities.
    """

    @property
    def categories(self) -> list[str]:
        """The sorted list of exposure category names."""
        # These need to be sorted so the cumulative sum is in the correct order of categories
        # and results are therefore reproducible and correct
        return sorted(self.exposure_params_table.value_columns)

    #####################
    # Lifecycle methods #
    #####################

    def __init__(
        self,
        causal_factor: EntityString,
        distribution_type: str,
        exposure_data: int | float | pd.DataFrame | None = None,
    ) -> None:
        """
        Parameters
        ----------
        causal_factor
            The entity string identifying the causal factor.
        distribution_type
            The distribution type (e.g., ``"ordered_polytomous"``).
        exposure_data
            Optional pre-loaded exposure data.
        """
        super().__init__(causal_factor, distribution_type, exposure_data)
        self.exposure_params_pipeline = f"{self.causal_factor}.exposure_parameters"

    #################
    # Setup methods #
    #################

    def setup(self, builder: Builder) -> None:
        """Build the exposure parameters table and register pipelines.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.
        """
        super().setup(builder)
        self.exposure_params_table = self.build_exposure_params_table(builder)
        self.register_exposure_params_pipeline(builder)

    def get_exposure_value_columns(
        self, exposure_data: int | float | pd.DataFrame
    ) -> list[str] | None:
        """Extract unique category names from exposure data.

        Parameters
        ----------
        exposure_data
            The exposure data, either as a scalar or a DataFrame.

        Returns
        -------
            A list of category names if the data is a DataFrame, or
            ``None`` for scalar data.
        """
        if isinstance(exposure_data, pd.DataFrame):
            return list(exposure_data["parameter"].unique())
        return None

    def register_exposure_ppf_pipeline(self, builder: Builder) -> None:
        """Register the polytomous exposure PPF pipeline.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.
        """
        builder.value.register_attribute_producer(
            self.exposure_ppf_pipeline,
            source=self.exposure_ppf,
            required_resources=[self.exposure_params_pipeline, self.causal_factor_propensity],
        )

    def register_exposure_params_pipeline(self, builder: Builder) -> None:
        """Register the exposure parameters pipeline.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.
        """
        builder.value.register_attribute_producer(
            self.exposure_params_pipeline, source=self.exposure_params_table
        )

    def build_exposure_params_table(self, builder: "Builder") -> LookupTable:
        """Build the lookup table for exposure parameters.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.

        Returns
        -------
            A lookup table for the exposure parameters.
        """
        data = self.get_exposure_data(builder)
        value_columns = self.get_exposure_value_columns(data)

        if isinstance(data, pd.DataFrame):
            data = pivot_categorical(data, "parameter")

        return self.build_lookup_table(
            builder, "exposure_parameters", data_source=data, value_columns=value_columns
        )

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def exposure_ppf(self, index: pd.Index) -> pd.Series:
        """Assign each simulant a category based on their propensity.

        Parameters
        ----------
        index
            An index representing the simulants.

        Returns
        -------
            A series of category labels for each simulant.

        Raises
        ------
        MissingDataError
            If all exposure data sums to zero.
        """
        pop = self.population_view.get(
            index, [self.causal_factor_propensity, self.exposure_params_pipeline]
        )
        quantiles = pop[self.causal_factor_propensity]
        sorted_exposures = pop[self.exposure_params_pipeline][self.categories]

        if not np.allclose(1, np.sum(sorted_exposures, axis=1)):
            raise MissingDataError("All exposure data returned as 0.")
        exposure_sum = sorted_exposures.cumsum(axis="columns")
        category_index = pd.concat(
            [exposure_sum[c] < quantiles for c in exposure_sum.columns], axis=1
        ).sum(axis=1)
        return pd.Series(
            np.array(self.categories)[category_index],
            name=self.causal_factor + ".exposure",
            index=quantiles.index,
        )


class DichotomousDistribution(CausalFactorDistribution):
    """Model risk exposure as a two-category (exposed/unexposed) distribution.

    Simulants with a propensity below the exposure probability are
    assigned to the exposed category; otherwise the unexposed category.
    Support optional rebinning of polytomous exposure data.
    """

    @property
    def exposed(self) -> str:
        """The name of the exposed category."""
        return "covered" if self.causal_factor.type == "intervention" else "exposed"

    @property
    def unexposed(self) -> str:
        """The name of the unexposed category."""
        return "uncovered" if self.causal_factor.type == "intervention" else "unexposed"

    def rename_deprecated_categories(self, data: pd.DataFrame) -> pd.DataFrame:
        """Rename deprecated cat1/cat2 parameter values to exposed/unexposed.

        If the data contains ``'cat1'`` in its ``'parameter'`` column, the
        values are replaced with the distribution's :attr:`exposed` and
        :attr:`unexposed` names.  A :class:`FutureWarning` is emitted for
        non-intervention causal factors to signal that the old names will be
        removed in a future release.

        Parameters
        ----------
        data
            A DataFrame with a ``'parameter'`` column.

        Returns
        -------
            The DataFrame with renamed parameter values (modified in place).
        """
        if "cat1" not in data["parameter"].values:
            return data

        if self.causal_factor.type != "intervention":
            warnings.warn(
                "Using 'cat1' and 'cat2' for dichotomous exposure is deprecated "
                "and will be removed in a future release. Use "
                f"'{self.exposed}' and '{self.unexposed}' instead.",
                FutureWarning,
                stacklevel=3,
            )
        data["parameter"] = data["parameter"].replace(
            {"cat1": self.exposed, "cat2": self.unexposed}
        )
        return data

    #####################
    # Lifecycle methods #
    #####################

    def __init__(
        self,
        causal_factor: EntityString,
        distribution_type: str,
        exposure_data: int | float | pd.DataFrame | None = None,
    ) -> None:
        """
        Parameters
        ----------
        causal_factor
            The entity string identifying the causal factor.
        distribution_type
            The distribution type (``"dichotomous"``).
        exposure_data
            Optional pre-loaded exposure data.
        """
        super().__init__(causal_factor, distribution_type, exposure_data)
        self.exposure_params_name = f"{self.causal_factor}.exposure_parameters"

    #################
    # Setup methods #
    #################

    def setup(self, builder: Builder) -> None:
        """Build the exposure table and register pipelines.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.
        """
        super().setup(builder)
        self.exposure_table = self.build_exposure_table(builder)
        self.register_exposure_params_pipeline(builder)

    def register_exposure_ppf_pipeline(self, builder: Builder) -> None:
        """Register the dichotomous exposure PPF pipeline.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.
        """
        builder.value.register_attribute_producer(
            self.exposure_ppf_pipeline,
            source=self.exposure_ppf,
            required_resources=[self.exposure_params_name, self.causal_factor_propensity],
        )

    def register_exposure_params_pipeline(self, builder: Builder) -> None:
        """Register the exposure parameters pipeline with calibration support.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.
        """
        register_risk_affected_attribute_producer(
            builder=builder,
            name=self.exposure_params_name,
            source=self.exposure_parameter_source,
            required_resources=[self.exposure_table],
        )

    def build_exposure_table(self, builder: Builder) -> LookupTable[pd.Series]:
        """Build a lookup table for exposure data.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.

        Returns
        -------
            A lookup table for the exposure data.

        Raises
        ------
        ValueError
            If any exposure values are outside the range [0, 1].
        """
        data = self.get_exposure_data(builder)

        if isinstance(data, pd.DataFrame):
            any_negatives = (data[DEFAULT_VALUE_COLUMN] < 0).any().any()
            any_over_one = (data[DEFAULT_VALUE_COLUMN] > 1).any().any()
            if any_negatives or any_over_one:
                raise ValueError(
                    f"All exposures must be in the range [0, 1] for {self.causal_factor}"
                )
        elif data < 0 or data > 1:
            raise ValueError(f"Exposure must be in the range [0, 1] for {self.causal_factor}")

        return self.build_lookup_table(builder, "exposure", data)

    def get_exposure_data(self, builder: Builder) -> int | float | pd.DataFrame:
        """Load and optionally rebin exposure data for the risk.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.

        Returns
        -------
            The (possibly rebinned) exposure data for the exposed
            category.
        """
        exposure_data = super().get_exposure_data(builder)

        if isinstance(exposure_data, (int, float)):
            return exposure_data

        # rebin exposure categories
        self.validate_rebin_source(builder, exposure_data)
        rebin_exposed_categories = set(self.configuration["rebinned_exposed"])
        exposure_data = self.rename_deprecated_categories(exposure_data)
        if rebin_exposed_categories:
            exposure_data = self._rebin_exposure_data(exposure_data, rebin_exposed_categories)

        exposure_data = exposure_data[exposure_data["parameter"] == self.exposed]
        return exposure_data.drop(columns="parameter")

    def _rebin_exposure_data(
        self, exposure_data: pd.DataFrame, rebin_exposed_categories: set
    ) -> pd.DataFrame:
        """Aggregate exposure categories into a single exposed category."""
        exposure_data = exposure_data[
            exposure_data["parameter"].isin(rebin_exposed_categories)
        ]
        exposure_data["parameter"] = self.exposed
        exposure_data = (
            exposure_data.groupby(list(exposure_data.columns.difference(["value"])))
            .sum()
            .reset_index()
        )
        return exposure_data

    ##############
    # Validators #
    ##############

    def validate_rebin_source(self, builder, data: pd.DataFrame) -> None:
        """Validate that rebinning configuration is consistent with the data.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.
        data
            The exposure data to validate against.

        Raises
        ------
        ValueError
            If rebinning and category thresholds are both specified,
            if any rebin categories are not found in the data, or if
            all categories are in the rebin set.
        """
        if not isinstance(data, pd.DataFrame):
            return

        rebin_exposed_categories = set(
            builder.configuration[self.causal_factor]["rebinned_exposed"]
        )

        if (
            rebin_exposed_categories
            and builder.configuration[self.causal_factor]["category_thresholds"]
        ):
            raise ValueError(
                f"Rebinning and category thresholds are mutually exclusive. "
                f"You provided both for {self.causal_factor.name}."
            )

        invalid_cats = rebin_exposed_categories.difference(set(data.parameter))
        if invalid_cats:
            raise ValueError(
                f"The following provided categories for the rebinned exposed "
                f"category of {self.causal_factor.name} are not found in the exposure data: "
                f"{invalid_cats}."
            )

        if rebin_exposed_categories == set(data.parameter):
            raise ValueError(
                f"The provided categories for the rebinned exposed category of "
                f"{self.causal_factor.name} comprise all categories for the exposure data. "
                f"At least one category must be left out of the provided categories "
                f"to be rebinned into the unexposed category."
            )

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def exposure_parameter_source(self, index: pd.Index) -> pd.Series:
        """Return exposure probabilities from the exposure lookup table.

        Parameters
        ----------
        index
            An index representing the simulants.

        Returns
        -------
            A series of exposure probabilities.
        """
        return self.exposure_table(index)

    def exposure_ppf(self, index: pd.Index) -> pd.Series:
        """Assign each simulant to the exposed or unexposed category based on propensity.

        Parameters
        ----------
        index
            An index representing the simulants.

        Returns
        -------
            A series of exposed or unexposed category labels.
        """
        pop = self.population_view.get(
            index, [self.causal_factor_propensity, self.exposure_params_name]
        )
        quantiles = pop[self.causal_factor_propensity]
        exposed = quantiles < pop[self.exposure_params_name]
        return pd.Series(
            exposed.replace({True: self.exposed, False: self.unexposed}),
            name=self.causal_factor + ".exposure",
            index=quantiles.index,
        )


def clip(q: pd.Series) -> pd.Series:
    """Clip quantile values to avoid distribution boundary issues.

    The risk distributions package uses the 99.9th and 0.001st percentiles
    of a log-normal distribution as the bounds of the distribution support.
    This is bound up in the GBD risk factor PAF calculation process.
    Clip the distribution tails so we don't get NaNs back from the
    distribution calls.

    Parameters
    ----------
    q
        A series of quantile values to clip.

    Returns
    -------
        The clipped quantile values.
    """
    Q_LOWER_BOUND = 0.0011
    Q_UPPER_BOUND = 0.998
    q[q > Q_UPPER_BOUND] = Q_UPPER_BOUND
    q[q < Q_LOWER_BOUND] = Q_LOWER_BOUND
    return q


def get_risk_distribution_parameter(data: float | pd.DataFrame) -> float | pd.Series:
    """Convert risk distribution parameter data to a usable format.

    If the data is a DataFrame, set the non-value columns as the index
    and squeeze to a Series.  Drop a ``"parameter"`` column if its only
    value is ``"continuous"``.

    Parameters
    ----------
    data
        The raw parameter data, either a scalar float or a DataFrame.

    Returns
    -------
        The parameter as a float or a ``pd.Series`` indexed by
        demographic columns.
    """
    if isinstance(data, pd.DataFrame):
        # don't return parameter col in continuous and ensemble distribution
        # means to match standard deviation index
        if "parameter" in data.columns and set(data["parameter"]) == {"continuous"}:
            data = data.drop("parameter", axis=1)
        index = [col for col in data.columns if col != DEFAULT_VALUE_COLUMN]
        data = data.set_index(index).squeeze(axis=1)

    return data
