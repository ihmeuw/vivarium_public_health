"""
====================================
Low Birth Weight and Short Gestation
====================================

Low birth weight and short gestation (LBWSG) is a non-standard risk
implementation that has been used in several public health models.
"""
import pickle
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.lifecycle import LifeCycleError
from vivarium.framework.lookup import LookupTable
from vivarium.framework.population import PopulationView, SimulantData
from vivarium.framework.values import Pipeline

from vivarium_public_health.risks import Risk, RiskEffect
from vivarium_public_health.risks.data_transformations import (
    get_exposure_data,
    get_exposure_post_processor,
)
from vivarium_public_health.risks.distributions import PolytomousDistribution
from vivarium_public_health.utilities import EntityString, to_snake_case

CATEGORICAL = "categorical"
BIRTH_WEIGHT = "birth_weight"
GESTATIONAL_AGE = "gestational_age"


class LBWSGDistribution(PolytomousDistribution):
    configuration_defaults = {
        "lbwsg_distribution": {
            "age_column": "age",
            "sex_column": "sex",
            "year_column": "year",
        }
    }

    def __init__(self, exposure_data: pd.DataFrame = None):
        super().__init__(
            EntityString("risk_factor.low_birth_weight_and_short_gestation"), exposure_data
        )
        self.exposure_data = exposure_data

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
        self.config = builder.configuration.lbwsg_distribution
        self.exposure_data = self._get_exposure_data(builder)

        super().setup(builder)
        self.category_intervals = self._get_category_intervals(builder)

    def _get_exposure_data(self, builder: Builder) -> pd.DataFrame:
        if self.exposure_data is None:
            self.exposure_data = get_exposure_data(builder, self.risk)

        return self.exposure_data.rename(
            columns={
                "sex": self.config.sex_column,
                "age_start": f"{self.config.age_column}_start",
                "age_end": f"{self.config.age_column}_end",
                "year_start": f"{self.config.year_column}_start",
                "year_end": f"{self.config.year_column}_end",
            }
        )

    def get_exposure_parameters(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.exposure_parameters_pipeline_name,
            source=builder.lookup.build_table(
                self.exposure_data,
                key_columns=[self.config.sex_column],
                parameter_columns=[self.config.age_column, self.config.year_column],
            ),
            requires_columns=[
                self.config.sex_column,
                self.config.age_column,
            ],
        )

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

        If categorical propensity is provided rather than exposure, this
        function requires access to the low birth weight and short gestation
        categorical exposure parameters pipeline
        ("risk_factor.low_birth_weight_and_short_gestation.exposure_parameters").

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
    AXES = [BIRTH_WEIGHT, GESTATIONAL_AGE]

    def __init__(self):
        super().__init__("risk_factor.low_birth_weight_and_short_gestation")

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

    def _get_exposure_pipeline(self, builder: Builder) -> Pipeline:
        # Exposure only used on initialization; not being saved to avoid a cycle
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
            self.birth_exposure_pipeline_name(axis): get_pipeline(axis) for axis in self.AXES
        }

    def _get_population_view(self, builder: Builder) -> PopulationView:
        columns = [self.exposure_column_name(axis) for axis in self.AXES]
        return builder.population.get_view(columns)

    def _register_simulant_initializer(self, builder: Builder) -> None:
        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            creates_columns=[self.exposure_column_name(axis) for axis in self.AXES],
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
            for axis in self.AXES
        }
        self.population_view.update(pd.DataFrame(birth_exposures))

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def _get_birth_exposure(self, axis: str, index: pd.Index) -> pd.DataFrame:
        categorical_propensity = self.randomness.get_draw(index, additional_key=CATEGORICAL)
        continuous_propensity = self.randomness.get_draw(index, additional_key=axis)
        return self.exposure_distribution.single_axis_ppf(
            axis, continuous_propensity, categorical_propensity
        )

    def _get_current_exposure(self, index: pd.Index) -> pd.DataFrame:
        raise LifeCycleError(
            f"The {self.risk.name} exposure pipeline should not be called. You probably want to"
            f" refer directly one of the exposure columns. During simulant initialization the birth"
            f" exposure pipelines should be used instead."
        )


class LBWSGRiskEffect(RiskEffect):
    TMREL_BIRTH_WEIGHT_INTERVAL: pd.Interval = pd.Interval(3500.0, 4500.0)
    TMREL_GESTATIONAL_AGE_INTERVAL: pd.Interval = pd.Interval(38.0, 42.0)

    def __init__(self, target: str):
        super().__init__("risk_factor.low_birth_weight_and_short_gestation", target)

        self.lbwsg_exposure_column_names = [
            LBWSGRisk.exposure_column_name(axis) for axis in LBWSGRisk.AXES
        ]
        self.relative_risk_pipeline_name = (
            f"effect_of_{self.risk.name}_on_{self.target.name}.relative_risk"
        )

    def relative_risk_column_name(self, age_group_id) -> str:
        return (
            f"effect_of_{self.risk.name}_on_{age_group_id}_{self.target.name}_relative_risk"
        )

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.age_intervals = self._get_age_intervals(builder)
        self.rr_column_names = self._get_rr_column_names()

        super().setup(builder)

        self.population_view = self._get_population_view(builder)
        self.interpolator = self._get_interpolator(builder)

        self._register_simulant_initializer(builder)

    def _get_risk_exposure(self, builder: Builder) -> Callable[[pd.Index], pd.DataFrame]:
        def exposure(index: pd.Index) -> pd.DataFrame:
            return self.population_view.subview(self.lbwsg_exposure_column_names).get(index)

        return exposure

    def _get_target_modifier(
        self, builder: Builder
    ) -> Callable[[pd.Index, pd.Series], pd.Series]:
        def adjust_target(index: pd.Index, target: pd.Series) -> pd.Series:
            return target * self.relative_risk(index)

        return adjust_target

    def _register_target_modifier(self, builder: Builder) -> None:
        builder.value.register_value_modifier(
            self.target_pipeline_name,
            modifier=self.target_modifier,
            requires_columns=["age", "sex"],
        )

    def _get_age_intervals(self, builder: Builder) -> Dict[str, pd.Interval]:
        age_bins = builder.data.load("population.age_bins").set_index("age_start")
        exposure = builder.data.load(f"{self.risk}.exposure")
        exposure = exposure[exposure["age_end"] > 0]

        exposed_age_group_starts = (
            exposure.groupby("age_start")["value"].any().reset_index()["age_start"]
        )

        return {
            to_snake_case(age_bins.loc[age_start, "age_group_name"]): pd.Interval(
                age_start, age_bins.loc[age_start, "age_end"]
            )
            for age_start in exposed_age_group_starts
        }

    def _get_rr_column_names(self) -> List[str]:
        return [self.relative_risk_column_name(age_group) for age_group in self.age_intervals]

    def _get_relative_risk_source(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.relative_risk_pipeline_name,
            source=self._get_relative_risk,
            requires_columns=["age"] + self.rr_column_names,
        )

    def _get_population_attributable_fraction_source(self, builder: Builder) -> LookupTable:
        return builder.lookup.build_table(
            builder.data.load(f"{self.risk}.population_attributable_fraction"),
            key_columns=["sex"],
            parameter_columns=["age", "year"],
        )

    def _get_population_view(self, builder: Builder) -> PopulationView:
        return builder.population.get_view(
            ["age", "sex"] + self.rr_column_names + self.lbwsg_exposure_column_names
        )

    def _get_interpolator(self, builder: Builder) -> pd.Series:
        age_start_to_age_group_name_map = {
            interval.left: to_snake_case(age_group_name)
            for age_group_name, interval in self.age_intervals.items()
        }

        # get relative risk data for target
        interpolators = builder.data.load(f"{self.risk}.relative_risk_interpolator")
        interpolators = (
            # isolate RRs for target and drop non-neonatal age groups since they have RR == 1.0
            interpolators[
                interpolators["age_start"].isin(
                    [interval.left for interval in self.age_intervals.values()]
                )
            ]
            .drop(columns=["age_end", "year_start", "year_end"])
            .set_index(["sex", "value"])
            .apply(lambda row: (age_start_to_age_group_name_map[row["age_start"]]), axis=1)
            .rename("age_group_name")
            .reset_index()
            .set_index(["sex", "age_group_name"])
        )["value"]

        interpolators = interpolators.apply(lambda x: pickle.loads(bytes.fromhex(x)))
        return interpolators

    def _register_simulant_initializer(self, builder: Builder) -> None:
        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            creates_columns=self.rr_column_names,
            requires_columns=["sex"] + self.lbwsg_exposure_column_names,
        )

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        pop = self.population_view.subview(["sex"] + self.lbwsg_exposure_column_names).get(
            pop_data.index
        )
        birth_weight = pop[LBWSGRisk.exposure_column_name(BIRTH_WEIGHT)]
        gestational_age = pop[LBWSGRisk.exposure_column_name(GESTATIONAL_AGE)]

        is_male = pop["sex"] == "Male"
        is_tmrel = (self.TMREL_GESTATIONAL_AGE_INTERVAL.left <= gestational_age) & (
            self.TMREL_BIRTH_WEIGHT_INTERVAL.left <= birth_weight
        )

        def get_relative_risk_for_age_group(age_group: int) -> pd.Series:
            column_name = self.relative_risk_column_name(age_group)
            log_relative_risk = pd.Series(0.0, index=pop_data.index, name=column_name)

            male_interpolator = self.interpolator["Male", age_group]
            log_relative_risk[is_male & ~is_tmrel] = male_interpolator(
                gestational_age[is_male & ~is_tmrel],
                birth_weight[is_male & ~is_tmrel],
                grid=False,
            )

            female_interpolator = self.interpolator["Female", age_group]
            log_relative_risk[~is_male & ~is_tmrel] = female_interpolator(
                gestational_age[~is_male & ~is_tmrel],
                birth_weight[~is_male & ~is_tmrel],
                grid=False,
            )
            return np.exp(log_relative_risk)

        relative_risk_columns = [
            get_relative_risk_for_age_group(age_group) for age_group in self.age_intervals
        ]
        self.population_view.update(pd.concat(relative_risk_columns, axis=1))

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def _get_relative_risk(self, index: pd.Index) -> pd.Series:
        pop = self.population_view.get(index)
        relative_risk = pd.Series(1.0, index=index, name=self.relative_risk_pipeline_name)

        for age_group, interval in self.age_intervals.items():
            age_group_mask = (interval.left <= pop["age"]) & (pop["age"] < interval.right)
            relative_risk[age_group_mask] = pop.loc[
                age_group_mask, self.relative_risk_column_name(age_group)
            ]
        return relative_risk
