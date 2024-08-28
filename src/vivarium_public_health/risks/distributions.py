"""
=================================
Risk Exposure Distribution Models
=================================

This module contains tools for modeling several different risk
exposure distributions.

"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import risk_distributions as rd
from layered_config_tree import LayeredConfigTree
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.population import SimulantData
from vivarium.framework.values import Pipeline, list_combiner, union_post_processor

from vivarium_public_health.risks.data_transformations import pivot_categorical
from vivarium_public_health.utilities import EntityString, get_lookup_columns


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
        exposure_data: Optional[Union[int, float, pd.DataFrame]] = None,
    ) -> None:
        super().__init__()
        self.risk = risk
        self.distribution_type = distribution_type
        self._exposure_data = exposure_data

        self.parameters_pipeline_name = f"{self.risk}.exposure_parameters"

    #################
    # Setup methods #
    #################

    def get_configuration(self, builder: "Builder") -> Optional[LayeredConfigTree]:
        return builder.configuration[self.risk]

    @abstractmethod
    def build_all_lookup_tables(self, builder: "Builder") -> None:
        raise NotImplementedError

    def get_exposure_data(self, builder: Builder) -> Union[int, float, pd.DataFrame]:
        if self._exposure_data is not None:
            return self._exposure_data
        return self.get_data(builder, self.configuration["data_sources"]["exposure"])

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.exposure_parameters = self.get_exposure_parameter_pipeline(builder)
        if self.exposure_parameters.name != self.parameters_pipeline_name:
            raise ValueError(
                "Expected exposure parameters pipeline to be named "
                f"{self.parameters_pipeline_name}, "
                f"but found {self.exposure_parameters.name}."
            )

    @abstractmethod
    def get_exposure_parameter_pipeline(self, builder: Builder) -> Pipeline:
        raise NotImplementedError

    ##################
    # Public methods #
    ##################

    @abstractmethod
    def ppf(self, quantiles: pd.Series) -> pd.Series:
        raise NotImplementedError


class EnsembleDistribution(RiskExposureDistribution):
    ##############
    # Properties #
    ##############

    @property
    def columns_created(self) -> List[str]:
        return [self._propensity]

    @property
    def initialization_requirements(self) -> Dict[str, List[str]]:
        return {
            "requires_columns": [],
            "requires_values": [],
            "requires_streams": [self._propensity],
        }

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, risk: EntityString, distribution_type: str = "ensemble") -> None:
        super().__init__(risk, distribution_type)
        self._propensity = f"ensemble_propensity_{self.risk}"

    #################
    # Setup methods #
    #################

    def build_all_lookup_tables(self, builder: Builder) -> None:
        exposure_data = self.get_exposure_data(builder)
        standard_deviation = self.get_data(
            builder,
            self.configuration["data_sources"]["exposure_standard_deviation"],
        )
        weights_source = self.configuration["data_sources"]["ensemble_distribution_weights"]
        raw_weights = self.get_data(builder, weights_source)

        glnorm_mask = raw_weights["parameter"] == "glnorm"
        if np.any(raw_weights.loc[glnorm_mask, self.get_value_columns(weights_source)]):
            raise NotImplementedError("glnorm distribution is not supported")
        raw_weights = raw_weights[~glnorm_mask]

        distributions = list(raw_weights["parameter"].unique())

        raw_weights = pivot_categorical(
            builder, self.risk, raw_weights, pivot_column="parameter", reset_index=False
        )

        weights, parameters = rd.EnsembleDistribution.get_parameters(
            raw_weights,
            mean=get_risk_distribution_parameter(self.get_value_columns, exposure_data),
            sd=get_risk_distribution_parameter(self.get_value_columns, standard_deviation),
        )

        distribution_weights_table = self.build_lookup_table(
            builder, weights.reset_index(), distributions
        )
        self.lookup_tables["ensemble_distribution_weights"] = distribution_weights_table
        key_columns = distribution_weights_table.key_columns
        parameter_columns = distribution_weights_table.parameter_columns

        self.parameters = {
            parameter: builder.lookup.build_table(
                data.reset_index(),
                key_columns=key_columns,
                parameter_columns=parameter_columns,
            )
            for parameter, data in parameters.items()
        }

    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self.randomness = builder.randomness.get_stream(self._propensity)

    def get_exposure_parameter_pipeline(self, builder: Builder) -> Pipeline:
        # This pipeline is not needed for ensemble distributions, so just
        # register a dummy pipeline
        def raise_not_implemented():
            raise NotImplementedError(
                "EnsembleDistribution does not use exposure parameters."
            )

        return builder.value.register_value_producer(
            self.parameters_pipeline_name, lambda *_: raise_not_implemented()
        )

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        ensemble_propensity = self.randomness.get_draw(pop_data.index).rename(
            self._propensity
        )
        self.population_view.update(ensemble_propensity)

    ##################
    # Public methods #
    ##################

    def ppf(self, quantiles: pd.Series) -> pd.Series:
        if not quantiles.empty:
            quantiles = clip(quantiles)
            weights = self.lookup_tables["ensemble_distribution_weights"](quantiles.index)
            parameters = {
                name: param(quantiles.index) for name, param in self.parameters.items()
            }
            ensemble_propensity = self.population_view.get(quantiles.index).iloc[:, 0]
            x = rd.EnsembleDistribution(weights, parameters).ppf(
                quantiles, ensemble_propensity
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

    #################
    # Setup methods #
    #################

    def build_all_lookup_tables(self, builder: "Builder") -> None:
        exposure_data = self.get_exposure_data(builder)
        standard_deviation = self.get_data(
            builder, self.configuration["data_sources"]["exposure_standard_deviation"]
        )
        parameters = self._distribution.get_parameters(
            mean=get_risk_distribution_parameter(self.get_value_columns, exposure_data),
            sd=get_risk_distribution_parameter(self.get_value_columns, standard_deviation),
        )

        self.lookup_tables["parameters"] = self.build_lookup_table(
            builder, parameters.reset_index(), list(parameters.columns)
        )

    def get_exposure_parameter_pipeline(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.parameters_pipeline_name,
            source=self.lookup_tables["parameters"],
            requires_columns=get_lookup_columns([self.lookup_tables["parameters"]]),
        )

    ##################
    # Public methods #
    ##################

    def ppf(self, quantiles: pd.Series) -> pd.Series:
        if not quantiles.empty:
            quantiles = clip(quantiles)
            parameters = self.exposure_parameters(quantiles.index)
            x = self._distribution(parameters=parameters).ppf(quantiles)
            x[x.isnull()] = 0
        else:
            x = pd.Series([])
        return x


class PolytomousDistribution(RiskExposureDistribution):
    @property
    def categories(self) -> List[str]:
        # These need to be sorted so the cumulative sum is in the ocrrect order of categories
        # and results are therefore reproducible and correct
        return sorted(self.lookup_tables["exposure"].value_columns)

    #################
    # Setup methods #
    #################

    def build_all_lookup_tables(self, builder: "Builder") -> None:
        exposure_data = self.get_exposure_data(builder)
        exposure_value_columns = self.get_exposure_value_columns(exposure_data)

        if isinstance(exposure_data, pd.DataFrame):
            exposure_data = pivot_categorical(builder, self.risk, exposure_data, "parameter")

        self.lookup_tables["exposure"] = self.build_lookup_table(
            builder, exposure_data, exposure_value_columns
        )

    def get_exposure_value_columns(
        self, exposure_data: Union[int, float, pd.DataFrame]
    ) -> Optional[List[str]]:
        if isinstance(exposure_data, pd.DataFrame):
            return list(exposure_data["parameter"].unique())
        return None

    def get_exposure_parameter_pipeline(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.parameters_pipeline_name,
            source=self.lookup_tables["exposure"],
            requires_columns=get_lookup_columns([self.lookup_tables["exposure"]]),
        )

    ##################
    # Public methods #
    ##################

    def ppf(self, quantiles: pd.Series) -> pd.Series:
        exposure = self.exposure_parameters(quantiles.index)
        sorted_exposures = exposure[self.categories]
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

    #################
    # Setup methods #
    #################

    def build_all_lookup_tables(self, builder: "Builder") -> None:
        exposure_data = self.get_exposure_data(builder)
        exposure_value_columns = self.get_exposure_value_columns(exposure_data)

        if isinstance(exposure_data, pd.DataFrame):
            any_negatives = (exposure_data[exposure_value_columns] < 0).any().any()
            any_over_one = (exposure_data[exposure_value_columns] > 1).any().any()
            if any_negatives or any_over_one:
                raise ValueError(f"All exposures must be in the range [0, 1] for {self.risk}")
        elif exposure_data < 0 or exposure_data > 1:
            raise ValueError(f"Exposure must be in the range [0, 1] for {self.risk}")

        self.lookup_tables["exposure"] = self.build_lookup_table(
            builder, exposure_data, exposure_value_columns
        )
        self.lookup_tables["paf"] = self.build_lookup_table(builder, 0.0)

    def get_exposure_data(self, builder: Builder) -> Union[int, float, pd.DataFrame]:
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

    def get_exposure_value_columns(
        self, exposure_data: Union[int, float, pd.DataFrame]
    ) -> Optional[List[str]]:
        if isinstance(exposure_data, pd.DataFrame):
            return self.get_value_columns(exposure_data)
        return None

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self.joint_paf = builder.value.register_value_producer(
            f"{self.risk}.exposure_parameters.paf",
            source=lambda index: [self.lookup_tables["paf"](index)],
            preferred_combiner=list_combiner,
            preferred_post_processor=union_post_processor,
        )

    def get_exposure_parameter_pipeline(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            f"{self.risk}.exposure_parameters",
            source=self.exposure_parameter_source,
            requires_columns=get_lookup_columns([self.lookup_tables["exposure"]]),
        )

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
        base_exposure = self.lookup_tables["exposure"](index).values
        joint_paf = self.joint_paf(index).values
        return pd.Series(base_exposure * (1 - joint_paf), index=index, name="values")

    ##################
    # Public methods #
    ##################

    def ppf(self, quantiles: pd.Series) -> pd.Series:
        exposed = quantiles < self.exposure_parameters(quantiles.index)
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


def get_risk_distribution_parameter(
    value_columns_getter: Callable[[Union[pd.DataFrame]], List[str]],
    data: Union[float, pd.DataFrame],
) -> Union[float, pd.Series]:
    if isinstance(data, pd.DataFrame):
        value_columns = value_columns_getter(data)
        if len(value_columns) > 1:
            raise ValueError(
                "Expected a single value column for risk data, but found "
                f"{len(value_columns)}: {value_columns}."
            )
        # don't return parameter col in continuous and ensemble distribution
        # means to match standard deviation index
        if "parameter" in data.columns and set(data["parameter"]) == {"continuous"}:
            data = data.drop("parameter", axis=1)
        index = [col for col in data.columns if col not in value_columns]
        data = data.set_index(index)[value_columns].squeeze(axis=1)

    return data
