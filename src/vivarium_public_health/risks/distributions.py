"""
=================================
Risk Exposure Distribution Models
=================================

This module contains tools for modeling several different risk
exposure distributions.

"""
from typing import Dict, List

import numpy as np
import pandas as pd
from risk_distributions import EnsembleDistribution, LogNormal, Normal
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.population import SimulantData
from vivarium.framework.values import Pipeline, list_combiner, union_post_processor

from vivarium_public_health.risks.data_transformations import get_distribution_data
from vivarium_public_health.utilities import EntityString


class MissingDataError(Exception):
    pass


# FIXME: This is a hack.  It's wrapping up an adaptor pattern in another
# adaptor pattern, which is gross, but would require some more difficult
# refactoring which is thoroughly out of scope right now. -J.C. 8/25/19
class SimulationDistribution(Component):
    """Wrapper around a variety of distribution implementations."""

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, risk: str):
        super().__init__()
        self.risk = EntityString(risk)

    def setup(self, builder: Builder) -> None:
        distribution_data = get_distribution_data(builder, self.risk)
        self.implementation = get_distribution(self.risk, **distribution_data)
        self.implementation.setup_component(builder)

    ##################
    # Public methods #
    ##################

    def ppf(self, q):
        return self.implementation.ppf(q)


class EnsembleSimulation(Component):
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

    def __init__(self, risk, weights, mean, sd):
        super().__init__()
        self.risk = EntityString(risk)
        self._weights, self._parameters = self.get_parameters(weights, mean, sd)
        self._propensity = f"ensemble_propensity_{self.risk}"

    def setup(self, builder: Builder) -> None:
        self.weights = builder.lookup.build_table(
            self._weights, key_columns=["sex"], parameter_columns=["age", "year"]
        )
        self.parameters = {
            k: builder.lookup.build_table(
                v, key_columns=["sex"], parameter_columns=["age", "year"]
            )
            for k, v in self._parameters.items()
        }

        self.randomness = builder.randomness.get_stream(self._propensity)

    ##########################
    # Initialization methods #
    ##########################

    def get_parameters(self, weights, mean, sd):
        index_cols = ["sex", "age_start", "age_end", "year_start", "year_end"]
        weights = weights.set_index(index_cols)
        mean = mean.set_index(index_cols)["value"]
        sd = sd.set_index(index_cols)["value"]
        weights, parameters = EnsembleDistribution.get_parameters(weights, mean=mean, sd=sd)
        return weights.reset_index(), {
            name: p.reset_index() for name, p in parameters.items()
        }

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

    def ppf(self, q):
        if not q.empty:
            q = clip(q)
            weights = self.weights(q.index)
            parameters = {
                name: parameter(q.index) for name, parameter in self.parameters.items()
            }
            ensemble_propensity = self.population_view.get(q.index).iloc[:, 0]
            x = EnsembleDistribution(weights, parameters).ppf(q, ensemble_propensity)
            x[x.isnull()] = 0
        else:
            x = pd.Series([])
        return x


class ContinuousDistribution(Component):
    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, risk, mean, sd, distribution=None):
        super().__init__()
        self.risk = EntityString(risk)
        self._distribution = distribution
        self._parameters = self.get_parameters(mean, sd)

    def setup(self, builder: Builder) -> None:
        self.parameters = builder.lookup.build_table(
            self._parameters, key_columns=["sex"], parameter_columns=["age", "year"]
        )

    ##########################
    # Initialization methods #
    ##########################

    def get_parameters(self, mean, sd):
        index = ["sex", "age_start", "age_end", "year_start", "year_end"]
        mean = mean.set_index(index)["value"]
        sd = sd.set_index(index)["value"]
        return self._distribution.get_parameters(mean=mean, sd=sd).reset_index()

    ##################
    # Public methods #
    ##################

    def ppf(self, q):
        if not q.empty:
            q = clip(q)
            x = self._distribution(parameters=self.parameters(q.index)).ppf(q)
            x[x.isnull()] = 0
        else:
            x = pd.Series([])
        return x


class PolytomousDistribution(Component):
    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, risk: str, exposure_data: pd.DataFrame):
        super().__init__()
        self.risk = EntityString(risk)
        self._exposure_data = exposure_data
        self.exposure_parameters_pipeline_name = f"{self.risk}.exposure_parameters"

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.categories = self.get_categories()
        self.exposure = self.get_exposure_parameters(builder)

    #################
    # Setup methods #
    #################

    def get_categories(self) -> List[str]:
        return sorted(
            [column for column in self._exposure_data if "cat" in column],
            key=lambda column: int(column[3:]),
        )

    def get_exposure_parameters(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.exposure_parameters_pipeline_name,
            source=builder.lookup.build_table(
                self._exposure_data,
                key_columns=["sex"],
                parameter_columns=["age", "year"],
            ),
        )

    ##################
    # Public methods #
    ##################

    def ppf(self, x: pd.Series) -> pd.Series:
        exposure = self.exposure(x.index)
        sorted_exposures = exposure[self.categories]
        if not np.allclose(1, np.sum(sorted_exposures, axis=1)):
            raise MissingDataError("All exposure data returned as 0.")
        exposure_sum = sorted_exposures.cumsum(axis="columns")
        category_index = pd.concat(
            [exposure_sum[c] < x for c in exposure_sum.columns], axis=1
        ).sum(axis=1)
        return pd.Series(
            np.array(self.categories)[category_index],
            name=self.risk + ".exposure",
            index=x.index,
        )


class DichotomousDistribution(Component):
    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, risk: str, exposure_data: pd.DataFrame):
        super().__init__()
        self.risk = risk
        self._exposure_data = exposure_data.drop(columns="cat2")

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self._base_exposure = builder.lookup.build_table(
            self._exposure_data, key_columns=["sex"], parameter_columns=["age", "year"]
        )
        self.exposure_proportion = builder.value.register_value_producer(
            f"{self.risk}.exposure_parameters", source=self.exposure
        )
        base_paf = builder.lookup.build_table(0)
        self.joint_paf = builder.value.register_value_producer(
            f"{self.risk}.exposure_parameters.paf",
            source=lambda index: [base_paf(index)],
            preferred_combiner=list_combiner,
            preferred_post_processor=union_post_processor,
        )

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def exposure(self, index: pd.Index) -> pd.Series:
        base_exposure = self._base_exposure(index).values
        joint_paf = self.joint_paf(index).values
        return pd.Series(base_exposure * (1 - joint_paf), index=index, name="values")

    ##################
    # Public methods #
    ##################

    def ppf(self, x: pd.Series) -> pd.Series:
        exposed = x < self.exposure_proportion(x.index)
        return pd.Series(
            exposed.replace({True: "cat1", False: "cat2"}),
            name=self.risk + ".exposure",
            index=x.index,
        )


def get_distribution(risk, distribution_type, exposure, exposure_standard_deviation, weights):
    if distribution_type == "dichotomous":
        distribution = DichotomousDistribution(risk, exposure)
    elif "polytomous" in distribution_type:
        distribution = PolytomousDistribution(risk, exposure)
    elif distribution_type == "normal":
        distribution = ContinuousDistribution(
            risk, mean=exposure, sd=exposure_standard_deviation, distribution=Normal
        )
    elif distribution_type == "lognormal":
        distribution = ContinuousDistribution(
            risk, mean=exposure, sd=exposure_standard_deviation, distribution=LogNormal
        )
    elif distribution_type == "ensemble":
        distribution = EnsembleSimulation(
            risk,
            weights,
            mean=exposure,
            sd=exposure_standard_deviation,
        )
    else:
        raise NotImplementedError(f"Unhandled distribution type {distribution_type}")
    return distribution


def clip(q):
    """Adjust the percentile boundary casses.

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
