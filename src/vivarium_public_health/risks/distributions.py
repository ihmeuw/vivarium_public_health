"""
=================================
Risk Exposure Distribution Models
=================================

This module contains tools for modeling several different risk
exposure distributions.

"""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np
import pandas as pd
import risk_distributions as rd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.population import SimulantData
from vivarium.framework.values import Pipeline, list_combiner, union_post_processor

from vivarium_public_health.risks.data_transformations import get_distribution_data
from vivarium_public_health.utilities import get_lookup_columns

if TYPE_CHECKING:
    from vivarium_public_health.risks import Risk


class MissingDataError(Exception):
    pass


class RiskExposureDistribution(Component, ABC):

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, risk_component: "Risk") -> None:
        super().__init__()
        self._risk_component = risk_component
        self.risk = self._risk_component.risk

        self.parameters_pipeline_name = f"{self.risk}.exposure_parameters"

    # noinspection PyAttributeOutsideInit
    def setup_component(self, builder: "Builder") -> None:
        distribution_data = get_distribution_data(builder, self._risk_component)
        self.exposure_data = distribution_data["exposure"]
        self.exposure_value_columns = distribution_data["exposure_value_columns"]
        self.standard_deviation = distribution_data["exposure_standard_deviation"]
        self.weights = distribution_data["weights"]
        super().setup_component(builder)

    #################
    # Setup methods #
    #################

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

    def __init__(self, risk: "Risk") -> None:
        super().__init__(risk)
        self._propensity = f"ensemble_propensity_{self.risk}"

    ##########################
    # Initialization methods #
    ##########################

    def build_all_lookup_tables(self, builder: Builder) -> None:
        raw_weights, distributions = self.weights
        weights, parameters = self.get_parameters(builder, raw_weights, distributions)

        distribution_weights_table = self.build_lookup_table(builder, weights, distributions)
        self.lookup_tables["ensemble_distribution_weights"] = distribution_weights_table
        key_columns = distribution_weights_table.key_columns
        parameter_columns = distribution_weights_table.parameter_columns

        self.parameters = {
            parameter: builder.lookup.build_table(
                data, key_columns=key_columns, parameter_columns=parameter_columns
            )
            for parameter, data in parameters.items()
        }

    def get_parameters(
        self, builder: Builder, raw_weights: pd.DataFrame, distributions: List[str]
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        value_columns = builder.data.value_columns()(f"{self.risk}.exposure")
        index_cols = [column for column in raw_weights.columns if column not in distributions]

        raw_weights = raw_weights.set_index(index_cols)
        if isinstance(self.exposure_data, pd.DataFrame):
            self.exposure_data = self.exposure_data.set_index(index_cols)[value_columns].squeeze(axis=1)
        if isinstance(self.standard_deviation, pd.DataFrame):
            self.standard_deviation = self.standard_deviation.set_index(index_cols)[value_columns].squeeze(axis=1)

        weights, parameters = rd.EnsembleDistribution.get_parameters(
            raw_weights, mean=self.exposure_data, sd=self.standard_deviation
        )
        weights = weights.reset_index()
        parameters = {name: p.reset_index() for name, p in parameters.items()}
        return weights, parameters

    #################
    # Setup methods #
    #################

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

    def __init__(self, risk: "Risk") -> None:
        super().__init__(risk)
        self._distribution = {
            "normal": rd.Normal,
            "lognormal": rd.LogNormal,
        }[risk.distribution_type]

    #################
    # Setup methods #
    #################

    def build_all_lookup_tables(self, builder: "Builder") -> None:
        value_columns = builder.data.value_columns()(f"{self.risk}.exposure")
        index = [col for col in self.exposure_data.columns if col not in value_columns]

        if isinstance(self.exposure_data, pd.DataFrame):
            self.exposure_data = self.exposure_data.set_index(index)[value_columns].squeeze(axis=1)
        if isinstance(self.standard_deviation, pd.DataFrame):
            self.standard_deviation = self.standard_deviation.set_index(index)[value_columns].squeeze(axis=1)
        parameters = self._distribution.get_parameters(mean=self.exposure_data, sd=self.standard_deviation)

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
            x = self._distribution(parameters=self.exposure_parameters(quantiles.index)).ppf(
                quantiles
            )
            x[x.isnull()] = 0
        else:
            x = pd.Series([])
        return x


class PolytomousDistribution(RiskExposureDistribution):
    @property
    def categories(self) -> List[str]:
        return self.lookup_tables["exposure"].value_columns

    #################
    # Setup methods #
    #################

    def build_all_lookup_tables(self, builder: "Builder") -> None:
        self.lookup_tables["exposure"] = self.build_lookup_table(
            builder, self.exposure_data, self.exposure_value_columns
        )

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
        if isinstance(self.exposure_data, pd.DataFrame):
            any_negatives = (self.exposure_data[self.exposure_value_columns] < 0).any().any()
            any_over_one = (self.exposure_data[self.exposure_value_columns] > 1).any().any()
            if any_negatives or any_over_one:
                raise ValueError(f"All exposures must be in the range [0, 1] for {self.risk}")
        elif self.exposure_data < 0 or self.exposure_data > 1:
            raise ValueError(f"Exposure must be in the range [0, 1] for {self.risk}")

        self.lookup_tables["exposure"] = self.build_lookup_table(
            builder, self.exposure_data, self.exposure_value_columns
        )
        self.lookup_tables["paf"] = self.build_lookup_table(builder, 0.0)

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


RISK_EXPOSURE_DISTRIBUTIONS = {
    "dichotomous": DichotomousDistribution,
    "ordered_polytomous": PolytomousDistribution,
    "unordered_polytomous": PolytomousDistribution,
    "normal": ContinuousDistribution,
    "lognormal": ContinuousDistribution,
    "ensemble": EnsembleDistribution,
}


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
