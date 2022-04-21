"""
=================================
Risk Exposure Distribution Models
=================================

This module contains tools for modeling several different risk
exposure distributions.

"""
from typing import Dict

import numpy as np
import pandas as pd
from risk_distributions import EnsembleDistribution, LogNormal, Normal
from vivarium.framework.engine import Builder
from vivarium.framework.values import list_combiner, union_post_processor

from vivarium_public_health.risks.data_transformations import get_distribution_data
from vivarium_public_health.utilities import EntityString


class MissingDataError(Exception):
    pass


# FIXME: This is a hack.  It's wrapping up an adaptor pattern in another
# adaptor pattern, which is gross, but would require some more difficult
# refactoring which is thorougly out of scope right now. -J.C. 8/25/19
class SimulationDistribution:
    """Wrapper around a variety of distribution implementations."""

    def __init__(self, risk):
        self.risk = risk

    @property
    def name(self):
        return f"{self.risk}.exposure_distribution"

    def setup(self, builder):
        distribution_data = get_distribution_data(builder, self.risk)
        self.implementation = get_distribution(self.risk, **distribution_data)
        self.implementation.setup(builder)

    def ppf(self, q):
        return self.implementation.ppf(q)

    def __repr__(self):
        return f"ExposureDistribution({self.risk})"


class EnsembleSimulation:
    def __init__(self, risk, weights, mean, sd):
        self.risk = risk
        self._weights, self._parameters = self._get_parameters(weights, mean, sd)

    @property
    def name(self):
        return f"ensemble_simulation.{self.risk}"

    def setup(self, builder):
        self.weights = builder.lookup.build_table(
            self._weights, key_columns=["sex"], parameter_columns=["age", "year"]
        )
        self.parameters = {
            k: builder.lookup.build_table(
                v, key_columns=["sex"], parameter_columns=["age", "year"]
            )
            for k, v in self._parameters.items()
        }

        self._propensity = f"ensemble_propensity_{self.risk}"
        self.randomness = builder.randomness.get_stream(self._propensity)

        self.population_view = builder.population.get_view([self._propensity])

        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            creates_columns=[self._propensity],
            requires_streams=[self._propensity],
        )

    def on_initialize_simulants(self, pop_data):
        ensemble_propensity = self.randomness.get_draw(pop_data.index).rename(
            self._propensity
        )
        self.population_view.update(ensemble_propensity)

    def _get_parameters(self, weights, mean, sd):
        index_cols = ["sex", "age_start", "age_end", "year_start", "year_end"]
        weights = weights.set_index(index_cols)
        mean = mean.set_index(index_cols)["value"]
        sd = sd.set_index(index_cols)["value"]
        weights, parameters = EnsembleDistribution.get_parameters(weights, mean=mean, sd=sd)
        return weights.reset_index(), {
            name: p.reset_index() for name, p in parameters.items()
        }

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

    def __repr__(self):
        return f"EnsembleSimulation(risk={self.risk})"


class ContinuousDistribution:
    def __init__(self, risk, mean, sd, distribution=None):
        self.risk = risk
        self.distribution = distribution
        self._parameters = self._get_parameters(mean, sd)

    @property
    def name(self):
        return f"simulation_distribution.{self.risk}"

    def setup(self, builder):
        self.parameters = builder.lookup.build_table(
            self._parameters, key_columns=["sex"], parameter_columns=["age", "year"]
        )

    def _get_parameters(self, mean, sd):
        index = ["sex", "age_start", "age_end", "year_start", "year_end"]
        mean = mean.set_index(index)["value"]
        sd = sd.set_index(index)["value"]
        return self.distribution.get_parameters(mean=mean, sd=sd).reset_index()

    def ppf(self, q):
        if not q.empty:
            q = clip(q)
            x = self.distribution(parameters=self.parameters(q.index)).ppf(q)
            x[x.isnull()] = 0
        else:
            x = pd.Series([])
        return x

    def __repr__(self):
        return f"SimulationDistribution(risk={self.risk}, distribution={self.distribution.__name__.lower()})"


class PolytomousDistribution:
    def __init__(self, risk: str, exposure_data: pd.DataFrame):
        self.risk = risk
        self.exposure_data = exposure_data
        self.categories = sorted(
            [column for column in self.exposure_data if "cat" in column],
            key=lambda column: int(column[3:]),
        )

    @property
    def name(self):
        return f"polytomous_distribution.{self.risk}"

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder):
        self.exposure = builder.value.register_value_producer(
            f"{self.risk}.exposure_parameters",
            source=builder.lookup.build_table(
                self.exposure_data, key_columns=["sex"], parameter_columns=["age", "year"]
            ),
        )

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

    def __repr__(self):
        return f"PolytomousDistribution(risk={self.risk})"


class DichotomousDistribution:
    def __init__(self, risk: str, exposure_data: pd.DataFrame):
        self.risk = risk
        self.exposure_data = exposure_data.drop("cat2", axis=1)

    @property
    def name(self):
        return f"dichotomous_distribution.{self.risk}"

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder):
        self._base_exposure = builder.lookup.build_table(
            self.exposure_data, key_columns=["sex"], parameter_columns=["age", "year"]
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

    def exposure(self, index: pd.Index) -> pd.Series:
        base_exposure = self._base_exposure(index).values
        joint_paf = self.joint_paf(index).values
        return pd.Series(base_exposure * (1 - joint_paf), index=index, name="values")

    def ppf(self, x: pd.Series) -> pd.Series:
        exposed = x < self.exposure_proportion(x.index)
        return pd.Series(
            exposed.replace({True: "cat1", False: "cat2"}),
            name=self.risk + ".exposure",
            index=x.index,
        )

    def __repr__(self):
        return f"DichotomousDistribution(risk={self.risk})"


class LBWSGDistribution(PolytomousDistribution):

    # todo make column names configurable

    CATEGORICAL_PROPENSITY_COLUMN = "low_birth_weight_and_short_gestation_propensity"
    BIRTH_WEIGHT = 'birth_weight'
    GESTATIONAL_AGE = 'gestational_age'

    def __init__(self, exposure_data: pd.DataFrame):
        super().__init__(
            EntityString("risk_factor.low_birth_weight_and_short_gestation"), exposure_data
        )

    def __repr__(self):
        return f"LBWSGDistribution()"

    ##############
    # Properties #
    ##############

    @property
    def name(self):
        return "lbwsg_distribution"

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder):
        super().setup(builder)
        self.category_intervals = self._get_category_intervals(builder)

    def _get_category_intervals(self, builder: Builder) -> Dict[str, Dict[str, pd.Interval]]:
        """
        Gets the intervals for each category. It is a dictionary from the string
        "birth_weight" or "gestational_age" to a dictionary from the category
        name to the interval
        :param builder:
        :return:
        """
        categories = builder.data.load(f'{self.risk}.categories')
        category_intervals = {
            axis: {
                category: self._parse_description(axis, description)
                for category, description in categories.items()
            }
            for axis in [self.BIRTH_WEIGHT, self.GESTATIONAL_AGE]
        }
        return category_intervals

    ##################
    # Public methods #
    ##################

    def ppf(self, propensities: pd.DataFrame) -> pd.DataFrame:
        """
        Takes a DataFrame with three columns:
        'low_birth_weight_and_short_gestation_propensity',
         'birth_weight.propensity', and 'gestational_age.propensity' which
        contain each of those propensities for each simulant.

        Returns a DataFrame with two columns for birth-weight and gestational
        age exposures.

        :param propensities:
        :return:
        """

        axes = [self.BIRTH_WEIGHT, self.GESTATIONAL_AGE]

        def get_exposure_interval(category: str) -> pd.Series:
            return pd.Series([self.category_intervals[axis][category] for axis in axes], index=axes)

        categorical_exposure = super().ppf(propensities[self.CATEGORICAL_PROPENSITY_COLUMN])
        exposure_intervals = categorical_exposure.apply(get_exposure_interval)
        continuous_exposures = [
            self._get_continuous_exposure(propensities, exposure_intervals, axis)
            for axis in self.category_intervals
        ]
        return pd.concat(continuous_exposures, axis=1)

    ##################
    # Helper methods #
    ##################

    @staticmethod
    def _parse_description(axis: str, description: str) -> pd.Interval:
        """
        Parses a string corresponding to a low birth weight and short gestation
        category to an Interval
        :param axis:
        :param description:
        :return:
        """
        endpoints = {
            LBWSGDistribution.BIRTH_WEIGHT: [
                float(val) for val in description.split(', [')[1].split(')')[0].split(', ')
            ],
            LBWSGDistribution.GESTATIONAL_AGE: [
                float(val) for val in description.split('- [')[1].split(')')[0].split(', ')
            ],
        }[axis]
        return pd.Interval(*endpoints, closed="left")   # noqa

    @staticmethod
    def _get_continuous_exposure(
        propensities: pd.DataFrame,
        exposure_intervals: pd.DataFrame,
        axis: str
    ) -> pd.Series:
        """
        Gets continuous exposures from a categorical exposure and propensity for
        a specific axis (i.e. birth-weight or gestational age).
        :param propensities:
        :param exposure_intervals:
        :param axis:
        :return:
        """
        propensity = propensities[f"{axis}.propensity"]
        exposure_left = exposure_intervals[axis].apply(lambda interval: interval.left)
        exposure_right = exposure_intervals[axis].apply(lambda interval: interval.right)
        continuous_exposure = propensity * (exposure_right - exposure_left) + exposure_left
        continuous_exposure = continuous_exposure.rename(f"{axis}.exposure")
        return continuous_exposure


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
    elif distribution_type == "lbwsg":
        distribution = LBWSGDistribution(exposure)
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
