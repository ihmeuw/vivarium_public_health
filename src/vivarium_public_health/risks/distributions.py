"""
=================================
Risk Exposure Distribution Models
=================================

This module contains tools for modeling several different risk
exposure distributions.

"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
import pandas as pd
import risk_distributions as rd
from layered_config_tree import LayeredConfigTree
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.lookup import DEFAULT_VALUE_COLUMN, LookupTable
from vivarium.framework.population import SimulantData
from vivarium.framework.resource import Resource
from vivarium.framework.values import list_combiner, union_post_processor

from vivarium_public_health.risks.data_transformations import pivot_categorical
from vivarium_public_health.utilities import EntityString


class MissingDataError(Exception):
    pass


class RiskExposureDistribution(Component, ABC):

    #####################
    # Lifecycle methods #
    #####################

    def __init__(
        self,
        risk: EntityString,
        distribution_type: str,
        exposure_data: int | float | pd.DataFrame | None = None,
    ) -> None:
        super().__init__()
        self.risk = risk
        self.distribution_type = distribution_type
        self._exposure_data = exposure_data

        self.risk_propensity = f"{self.risk.name}.propensity"
        self.exposure_ppf_pipeline = f"{self.risk.name}.exposure_distribution.ppf"

    #################
    # Setup methods #
    #################

    def get_configuration(self, builder: "Builder") -> LayeredConfigTree | None:
        return builder.configuration[self.risk]

    def get_exposure_data(self, builder: Builder) -> int | float | pd.DataFrame:
        if self._exposure_data is not None:
            return self._exposure_data
        return self.get_data(builder, self.configuration["data_sources"]["exposure"])

    def setup(self, builder: Builder) -> None:
        self.register_exposure_ppf_pipeline(builder)

    @abstractmethod
    def register_exposure_ppf_pipeline(self, builder: Builder) -> None:
        pass


class EnsembleDistribution(RiskExposureDistribution):
    ##############
    # Properties #
    ##############

    @property
    def columns_created(self) -> list[str]:
        return [self.ensemble_propensity]

    @property
    def initialization_requirements(self) -> list[str | Resource]:
        return [self.randomness]

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, risk: EntityString, distribution_type: str = "ensemble") -> None:
        super().__init__(risk, distribution_type)
        self.ensemble_propensity = f"ensemble_propensity_{self.risk}"

    #################
    # Setup methods #
    #################

    def setup(self, builder: Builder) -> None:
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
                value_columns=[
                    *rd.EnsembleDistribution.distribution_map[parameter].expected_parameters,
                    "x_min",
                    "x_max",
                ],
            )
            for parameter, data in parameters.items()
        }

        super().setup(builder)
        self.randomness = builder.randomness.get_stream(
            self.ensemble_propensity, component=self
        )

    def get_distribution_definitions(
        self, builder: Builder
    ) -> tuple[list[str], pd.DataFrame, dict[str, pd.DataFrame]]:
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
        tables = [self.distribution_weights_table, *self.parameters.values()]

        builder.value.register_attribute_producer(
            self.exposure_ppf_pipeline,
            source=self.exposure_ppf,
            component=self,
            required_resources=[*tables, self.risk_propensity, self.ensemble_propensity],
        )

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        ensemble_propensity = self.randomness.get_draw(pop_data.index).rename(
            self.ensemble_propensity
        )
        self.population_view.update(ensemble_propensity)

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def exposure_ppf(self, index: pd.Index) -> pd.Series:
        pop = self.population_view.get_attributes(
            index, [self.risk.propensity_name, self.ensemble_propensity]
        )
        quantiles = pop[self.risk.propensity_name]

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


class ContinuousDistribution(RiskExposureDistribution):
    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, risk: EntityString, distribution_type: str) -> None:
        super().__init__(risk, distribution_type)
        self.exposure_params_name = f"{self.risk}.exposure_parameters"
        self.standard_deviation = None
        try:
            self._distribution = {
                "normal": rd.Normal,
                "lognormal": rd.LogNormal,
            }[distribution_type]
        except KeyError:
            raise NotImplementedError(
                f"Distribution type {distribution_type} is not supported for "
                f"risk {risk.name}."
            )

    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self.register_exposure_parameter_pipeline(builder)

    #################
    # Setup methods #
    #################

    def setup(self, builder):
        parameters = self.get_distribution_parameters(builder)
        self.parameters_table = self.build_lookup_table(
            builder,
            "exposure_parameters",
            data_source=parameters.reset_index(),
            value_columns=list(parameters.columns),
        )
        super().setup(builder)

    def get_distribution_parameters(self, builder: "Builder") -> None:
        exposure_data = self.get_exposure_data(builder)
        standard_deviation = self.get_data(
            builder, self.configuration["data_sources"]["exposure_standard_deviation"]
        )
        return self._distribution.get_parameters(
            mean=get_risk_distribution_parameter(exposure_data),
            sd=get_risk_distribution_parameter(standard_deviation),
        )

    def register_exposure_ppf_pipeline(self, builder: Builder) -> None:
        builder.value.register_attribute_producer(
            self.exposure_ppf_pipeline,
            source=self.exposure_ppf,
            component=self,
            required_resources=[self.exposure_params_name, self.risk_propensity],
        )

    def register_exposure_params_pipeline(self, builder: Builder) -> None:
        builder.value.register_attribute_producer(
            self.exposure_params_name, source=self.parameters_table, component=self
        )

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def exposure_ppf(self, index: pd.Index) -> pd.Series:
        pop = self.population_view.get_attributes(
            index, [self.risk_propensity, self.exposure_params_name]
        )
        quantiles = pop[self.risk_propensity]

        if not quantiles.empty:
            quantiles = clip(quantiles)
            x = self._distribution(parameters=pop[self.exposure_params_name]).ppf(quantiles)
            x[x.isnull()] = 0
        else:
            x = pd.Series([])
        return x


class PolytomousDistribution(RiskExposureDistribution):
    @property
    def categories(self) -> list[str]:
        # These need to be sorted so the cumulative sum is in the correct order of categories
        # and results are therefore reproducible and correct
        return sorted(self.exposure_params_table.value_columns)

    #####################
    # Lifecycle methods #
    #####################

    def __init__(
        self,
        risk: EntityString,
        distribution_type: str,
        exposure_data: int | float | pd.DataFrame | None = None,
    ) -> None:
        super().__init__(risk, distribution_type, exposure_data)
        self.exposure_params_pipeline = f"{self.risk}.exposure_parameters"

    #####################
    # Lifecycle methods #
    #####################

    def __init__(
        self,
        risk: EntityString,
        distribution_type: str,
        exposure_data: int | float | pd.DataFrame | None = None,
    ) -> None:
        super().__init__(risk, distribution_type, exposure_data)
        self.exposure_params_pipeline = f"{self.risk}.exposure_parameters"

    #################
    # Setup methods #
    #################

    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self.exposure_params_table = self.build_exposure_params_table(builder)
        self.register_exposure_params_pipeline(builder)

    def get_exposure_value_columns(
        self, exposure_data: int | float | pd.DataFrame
    ) -> list[str] | None:
        if isinstance(exposure_data, pd.DataFrame):
            return list(exposure_data["parameter"].unique())
        return None

    def register_exposure_ppf_pipeline(self, builder: Builder) -> None:
        builder.value.register_attribute_producer(
            self.exposure_ppf_pipeline,
            source=self.exposure_ppf,
            component=self,
            required_resources=[self.exposure_params_pipeline, self.risk_propensity],
        )

    def register_exposure_params_pipeline(self, builder: Builder) -> None:
        builder.value.register_attribute_producer(
            self.exposure_params_pipeline, source=self.exposure_params_table, component=self
        )

    def build_exposure_params_table(self, builder: "Builder"):
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
        pop = self.population_view.get_attributes(
            index, [self.risk_propensity, self.exposure_params_pipeline]
        )
        quantiles = pop[self.risk_propensity]
        sorted_exposures = pop[self.exposure_params_pipeline][self.categories]

        if not np.allclose(1, np.sum(sorted_exposures, axis=1)):
            raise MissingDataError("All exposure data returned as 0.")
        exposure_sum = sorted_exposures.cumsum(axis="columns")
        category_index = pd.concat(
            [exposure_sum[c] < quantiles for c in exposure_sum.columns], axis=1
        ).sum(axis=1)
        return pd.Series(
            np.array(self.categories)[category_index],
            name=self.risk + ".exposure",
            index=quantiles.index,
        )


class DichotomousDistribution(RiskExposureDistribution):

    #####################
    # Lifecycle methods #
    #####################

    def __init__(
        self,
        risk: EntityString,
        distribution_type: str,
        exposure_data: int | float | pd.DataFrame | None = None,
    ) -> None:
        super().__init__(risk, distribution_type, exposure_data)
        self.exposure_params_name = f"{self.risk}.exposure_parameters"
        self.exposure_params_paf_name = f"{self.exposure_params_name}.paf"

    #################
    # Setup methods #
    #################

    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self.exposure_table = self.build_exposure_table(builder)
        self.paf_table = self.build_lookup_table(builder, "exposure_paf", 0.0)
        self.register_exposure_params_pipeline(builder)
        builder.value.register_attribute_producer(
            self.exposure_params_paf_name,
            source=lambda index: [self.paf_table(index)],
            component=self,
            preferred_combiner=list_combiner,
            preferred_post_processor=union_post_processor,
        )

    def register_exposure_ppf_pipeline(self, builder: Builder) -> None:
        builder.value.register_attribute_producer(
            self.exposure_ppf_pipeline,
            source=self.exposure_ppf,
            component=self,
            required_resources=[self.exposure_params_name, self.risk_propensity],
        )

    def register_exposure_params_pipeline(self, builder: Builder) -> None:
        builder.value.register_attribute_producer(
            self.exposure_params_name,
            source=self.exposure_parameter_source,
            component=self,
            required_resources=[self.exposure_table],
        )

    def build_exposure_table(self, builder: Builder) -> LookupTable[pd.Series]:
        data = self.get_exposure_data(builder)

        if isinstance(data, pd.DataFrame):
            any_negatives = (data[DEFAULT_VALUE_COLUMN] < 0).any().any()
            any_over_one = (data[DEFAULT_VALUE_COLUMN] > 1).any().any()
            if any_negatives or any_over_one:
                raise ValueError(f"All exposures must be in the range [0, 1] for {self.risk}")
        elif data < 0 or data > 1:
            raise ValueError(f"Exposure must be in the range [0, 1] for {self.risk}")

        return self.build_lookup_table(builder, "exposure", data)

    def get_exposure_data(self, builder: Builder) -> int | float | pd.DataFrame:
        exposure_data = super().get_exposure_data(builder)

        if isinstance(exposure_data, (int, float)):
            return exposure_data

        # rebin exposure categories
        self.validate_rebin_source(builder, exposure_data)
        rebin_exposed_categories = set(self.configuration["rebinned_exposed"])
        if rebin_exposed_categories:
            exposure_data = self._rebin_exposure_data(exposure_data, rebin_exposed_categories)

        exposure_data = exposure_data[exposure_data["parameter"] == "cat1"]
        return exposure_data.drop(columns="parameter")

    @staticmethod
    def _rebin_exposure_data(
        exposure_data: pd.DataFrame, rebin_exposed_categories: set
    ) -> pd.DataFrame:
        exposure_data = exposure_data[
            exposure_data["parameter"].isin(rebin_exposed_categories)
        ]
        exposure_data["parameter"] = "cat1"
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
        if not isinstance(data, pd.DataFrame):
            return

        rebin_exposed_categories = set(builder.configuration[self.risk]["rebinned_exposed"])

        if (
            rebin_exposed_categories
            and builder.configuration[self.risk]["category_thresholds"]
        ):
            raise ValueError(
                f"Rebinning and category thresholds are mutually exclusive. "
                f"You provided both for {self.risk.name}."
            )

        invalid_cats = rebin_exposed_categories.difference(set(data.parameter))
        if invalid_cats:
            raise ValueError(
                f"The following provided categories for the rebinned exposed "
                f"category of {self.risk.name} are not found in the exposure data: "
                f"{invalid_cats}."
            )

        if rebin_exposed_categories == set(data.parameter):
            raise ValueError(
                f"The provided categories for the rebinned exposed category of "
                f"{self.risk.name} comprise all categories for the exposure data. "
                f"At least one category must be left out of the provided categories "
                f"to be rebinned into the unexposed category."
            )

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def exposure_parameter_source(self, index: pd.Index) -> pd.Series:
        base_exposure = self.exposure_table(index).values
        joint_paf = self.population_view.get_attributes(
            index, self.exposure_params_paf_name
        ).values
        return pd.Series(base_exposure * (1 - joint_paf), index=index, name="values")

    def exposure_ppf(self, index: pd.Index) -> pd.Series:
        pop = self.population_view.get_attributes(
            index, [self.risk_propensity, self.exposure_params_name]
        )
        quantiles = pop[self.risk_propensity]
        exposed = quantiles < pop[self.exposure_params_name]
        return pd.Series(
            exposed.replace({True: "cat1", False: "cat2"}),
            name=self.risk + ".exposure",
            index=quantiles.index,
        )


def clip(q):
    """Adjust the percentile boundary cases.

    The  risk distributions package uses the 99.9th and 0.001st percentiles
    of a log-normal distribution as the bounds of the distribution support.
    This is bound up in the GBD risk factor PAF calculation process.
    We'll clip the distribution tails so we don't get NaNs back from the
    distribution calls
    """
    Q_LOWER_BOUND = 0.0011
    Q_UPPER_BOUND = 0.998
    q[q > Q_UPPER_BOUND] = Q_UPPER_BOUND
    q[q < Q_LOWER_BOUND] = Q_LOWER_BOUND
    return q


def get_risk_distribution_parameter(data: float | pd.DataFrame) -> float | pd.Series:
    if isinstance(data, pd.DataFrame):
        # don't return parameter col in continuous and ensemble distribution
        # means to match standard deviation index
        if "parameter" in data.columns and set(data["parameter"]) == {"continuous"}:
            data = data.drop("parameter", axis=1)
        index = [col for col in data.columns if col != DEFAULT_VALUE_COLUMN]
        data = data.set_index(index).squeeze(axis=1)

    return data
