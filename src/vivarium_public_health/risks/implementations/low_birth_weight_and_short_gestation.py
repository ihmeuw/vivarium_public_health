"""
====================================
Low Birth Weight and Short Gestation
====================================

Low birth weight and short gestation (LBWSG) is a non-standard risk
implementation that has been used in several public health models.
"""
from typing import Dict

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.lifecycle import LifeCycleError
from vivarium.framework.population import PopulationView, SimulantData
from vivarium.framework.values import Pipeline

from vivarium_public_health.risks import Risk
from vivarium_public_health.risks.data_transformations import (
    get_exposure_data,
    get_exposure_post_processor,
)
from vivarium_public_health.risks.distributions import PolytomousDistribution
from vivarium_public_health.utilities import EntityString

CATEGORICAL = "categorical"
BIRTH_WEIGHT = "birth_weight"
GESTATIONAL_AGE = "gestational_age"


class LBWSGDistribution(PolytomousDistribution):

    def __init__(self, exposure_data: pd.DataFrame = None):
        super().__init__(
            EntityString("risk_factor.low_birth_weight_and_short_gestation"), exposure_data
        )

    def __repr__(self):
        return "LBWSGDistribution()"

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
        if self.exposure_data is None:
            self.exposure_data = get_exposure_data(builder, self.risk)

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
        categories = builder.data.load(f"{self.risk}.categories")
        category_intervals = {
            axis: {
                category: self._parse_description(axis, description)
                for category, description in categories.items()
            }
            for axis in [BIRTH_WEIGHT, GESTATIONAL_AGE]
        }
        return category_intervals

    ##################
    # Public methods #
    ##################

    def ppf(self, propensities: pd.DataFrame) -> pd.DataFrame:
        """
        Takes a DataFrame with three columns: 'categorical.propensity',
        'birth_weight.propensity', and 'gestational_age.propensity' which
        contain each of those propensities for each simulant.

        Returns a DataFrame with two columns for birth-weight and gestational
        age exposures.

        :param propensities:
        :return:
        """

        categorical_exposure = super().ppf(propensities[f"{CATEGORICAL}_propensity"])
        continuous_exposures = [
            self.single_axis_ppf(
                axis,
                propensities[f"{axis}.propensity"],
                categorical_exposure=categorical_exposure,
            )
            for axis in self.category_intervals
        ]
        return pd.concat(continuous_exposures, axis=1)

    def single_axis_ppf(
        self,
        axis: str,
        propensity: pd.Series,
        categorical_propensity: pd.Series = None,
        categorical_exposure: pd.Series = None,
    ) -> pd.Series:
        """
        Takes an axis (either 'birth_weight' or 'gestational_age'), a propensity
        and either a categorical propensity or a categorical exposure and
        returns continuous exposures for that axis.

        :param axis:
        :param propensity:
        :param categorical_propensity:
        :param categorical_exposure:
        :return:
        """

        if (categorical_propensity is None) == (categorical_exposure is None):
            raise ValueError(
                "Either categorical propensity of categorical exposure may be provided, but not"
                " both or neither."
            )

        if categorical_exposure is None:
            categorical_exposure = super().ppf(categorical_propensity)

        exposure_intervals = categorical_exposure.apply(
            lambda category: self.category_intervals[axis][category]
        )

        exposure_left = exposure_intervals.apply(lambda interval: interval.left)
        exposure_right = exposure_intervals.apply(lambda interval: interval.right)
        continuous_exposure = propensity * (exposure_right - exposure_left) + exposure_left
        continuous_exposure = continuous_exposure.rename(f"{axis}.exposure")
        return continuous_exposure

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
            BIRTH_WEIGHT: [
                float(val) for val in description.split(", [")[1].split(")")[0].split(", ")
            ],
            GESTATIONAL_AGE: [
                float(val) for val in description.split("- [")[1].split(")")[0].split(", ")
            ],
        }[axis]
        return pd.Interval(*endpoints, closed="left")  # noqa


class LBWSGRisk(Risk):
    def __init__(self):
        super().__init__("risk_factor.low_birth_weight_and_short_gestation")

        self.axes = [BIRTH_WEIGHT, GESTATIONAL_AGE]

    @staticmethod
    def propensity_column_name(axis: str) -> str:
        return f"{axis}_propensity"

    @staticmethod
    def propensity_pipeline_name(axis: str) -> str:
        return f"{axis}.propensity"

    @staticmethod
    def birth_exposure_pipeline_name(axis: str) -> str:
        return f"{axis}.birth_exposure"

    @staticmethod
    def exposure_column_name(axis: str) -> str:
        return f"{axis}_exposure"

    ##########################
    # Initialization methods #
    ##########################

    def _get_exposure_distribution(self) -> LBWSGDistribution:
        return LBWSGDistribution()

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder):
        super().setup(builder)

        self.birth_exposures = self._get_birth_exposure_pipelines(builder)

    def _get_propensity_pipeline(self, builder: Builder) -> Pipeline:
        # Propensity only used on initialization; not being saved to avoid a cycle
        return None

    def _get_birth_exposure_pipelines(self, builder: Builder) -> Dict[str, Pipeline]:
        def get_pipeline(axis_: str):
            return builder.value.register_value_producer(
                self.birth_exposure_pipeline_name(axis_),
                source=lambda index: self._get_birth_exposure(axis_, index),
                requires_columns=["age", "sex"],
                requires_streams=[self._randomness_stream_name],
                preferred_post_processor=get_exposure_post_processor(builder, self.risk),
            )

        return {
            self.birth_exposure_pipeline_name(axis): get_pipeline(axis) for axis in self.axes
        }

    def _get_population_view(self, builder: Builder) -> PopulationView:
        columns = [self.exposure_column_name(axis) for axis in self.axes]
        return builder.population.get_view(columns)

    def _register_simulant_initializer(self, builder: Builder) -> None:
        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            creates_columns=[self.exposure_column_name(axis) for axis in self.axes],
            requires_values=[self.exposure_column_name(axis) for axis in self.axes],
            requires_streams=[self._randomness_stream_name],
        )

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        birth_exposures = {
            self.exposure_column_name(axis): self.birth_exposures[
                self.birth_exposure_pipeline_name(axis)
            ](pop_data.index)
            for axis in self.axes
        }
        self.population_view.update(pd.DataFrame(birth_exposures))

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def _get_birth_exposure(self, axis: str, index: pd.Index) -> pd.DataFrame:
        categorical_propensity = self.randomness.get_draw(index, additional_key=CATEGORICAL)
        continuous_propensity = self.randomness.get_draw(index, additional_key=axis)
        # todo account for risk-specific shift (population-attributable-quantity) in risk effect
        return self.exposure_distribution.single_axis_ppf(
            axis, continuous_propensity, categorical_propensity
        )

    def _get_current_exposure(self, index: pd.Index) -> pd.DataFrame:
        raise LifeCycleError(
            f"The {self.risk.name} exposure pipeline should not be called. You probably want to"
            f" refer directly one of the exposure columns. During simulant initialization the birth"
            f" exposure pipelines should be used instead."
        )
