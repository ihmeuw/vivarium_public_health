"""
====================================
Low Birth Weight and Short Gestation
====================================

Low birth weight and short gestation (LBWSG) is a non-standard risk
implementation that has been used in several public health models.

"""
from __future__ import annotations

import pickle
import re
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from layered_config_tree import ConfigurationError
from loguru import logger
from vivarium.framework.engine import Builder
from vivarium.framework.lifecycle import LifeCycleError
from vivarium.framework.population import SimulantData
from vivarium.framework.resource import Resource
from vivarium.framework.values import Pipeline

from vivarium_public_health.risks import Risk, RiskEffect
from vivarium_public_health.risks.data_transformations import (
    get_exposure_post_processor,
    pivot_categorical,
)
from vivarium_public_health.risks.distributions import PolytomousDistribution
from vivarium_public_health.utilities import EntityString, get_lookup_columns, to_snake_case

CATEGORICAL = "categorical"
BIRTH_WEIGHT = "birth_weight"
GESTATIONAL_AGE = "gestational_age"
AXES = [BIRTH_WEIGHT, GESTATIONAL_AGE]


class LBWSGDistribution(PolytomousDistribution):
    @property
    def categories(self) -> list[str]:
        # These need to be sorted so the cumulative sum is in the correct order of categories
        # and results are therefore reproducible and correct
        return sorted(self.lookup_tables[self.exposure_key].value_columns)

    #################
    # Setup methods #
    #################

    def __init__(
        self,
        risk: EntityString,
        distribution_type: str,
        exposure_data: int | float | pd.DataFrame | None = None,
    ) -> None:
        super().__init__(risk, distribution_type, exposure_data)
        self.exposure_key = "birth_exposure"
        self.risk_propensity = f"{self.risk.name}.categorical_propensity"

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self.category_intervals = self.get_category_intervals(builder)

    def build_all_lookup_tables(self, builder: Builder) -> None:
        try:
            birth_exposure_data = self.get_data(
                builder, self.configuration["data_sources"]["birth_exposure"]
            )
            birth_exposure_value_columns = self.get_exposure_value_columns(
                birth_exposure_data
            )

            if isinstance(birth_exposure_data, pd.DataFrame):
                birth_exposure_data = pivot_categorical(
                    builder, self.risk, birth_exposure_data, "parameter"
                )

            self.lookup_tables["birth_exposure"] = self.build_lookup_table(
                builder, birth_exposure_data, birth_exposure_value_columns
            )
        except ConfigurationError:
            logger.warning("Birth exposure data for LBWSG is missing from the simulation")
        try:
            super().build_all_lookup_tables(builder)
        except ConfigurationError:
            logger.warning("The data for LBWSG exposure is missing from the simulation.")

        if (
            "birth_exposure" not in self.lookup_tables
            and "exposure" not in self.lookup_tables
        ):
            raise ConfigurationError(
                "The LBWSG distribution requires either 'birth_exposure' or 'exposure' data to be "
                "available in the simulation."
            )

    def register_ppf_pipeline(self, builder):
        required_resources = [self.exposure_params_name, self.risk_propensity] + [
            LBWSGRisk.get_continuous_propensity_name(axis) for axis in AXES
        ]
        builder.value.register_attribute_producer(
            self.ppf_pipeline,
            source=self.ppf,
            component=self,
            required_resources=required_resources,
        )

    def register_exposure_parameter_pipeline(self, builder: Builder) -> None:
        lookup_columns = []
        if "exposure" in self.lookup_tables:
            lookup_columns.extend(get_lookup_columns([self.lookup_tables["exposure"]]))
        if "birth_exposure" in self.lookup_tables:
            lookup_columns.extend(get_lookup_columns([self.lookup_tables["birth_exposure"]]))
        builder.value.register_attribute_producer(
            self.exposure_params_name,
            source=lambda index: self.lookup_tables[self.exposure_key](index),
            component=self,
            required_resources=list(set(lookup_columns)),
        )

    def get_category_intervals(self, builder: Builder) -> dict[str, dict[str, pd.Interval]]:
        """Gets the intervals for each category.

        Parameters
        ----------
        builder
            The builder object.

        Returns
        -------
            The intervals for each category.
        """
        categories: dict[str, str] = builder.data.load(f"{self.risk}.categories")
        category_intervals = {GESTATIONAL_AGE: {}, BIRTH_WEIGHT: {}}

        for category, description in categories.items():
            gestation_interval, birth_weight_interval = self._parse_description(description)
            category_intervals[GESTATIONAL_AGE][category] = gestation_interval
            category_intervals[BIRTH_WEIGHT][category] = birth_weight_interval
        return category_intervals

    ##################
    # Public methods #
    ##################

    def ppf(self, index: pd.Index) -> pd.DataFrame:
        """Calculate continuous exposures from propensities.

        Parameters
        ----------
        propensities
            Propensities DataFrame for each simulant with three columns:
            'categorical.propensity', 'birth_weight.propensity', and
            'gestational_age.propensity'.

        Returns
        -------
            A DataFrame with two columns for birth-weight and gestational age
            exposures.
        """
        propensities = self.population_view.get_attributes(
            index,
            [LBWSGRisk.get_continuous_propensity_name(axis) for axis in AXES],
        )

        categorical_exposure = super().ppf(index=propensities.index)
        continuous_exposures = {
            axis: self.single_axis_ppf(
                axis,
                propensities[LBWSGRisk.get_continuous_propensity_name(axis)],
                categorical_exposure=categorical_exposure,
            )
            for axis in AXES
        }
        return pd.DataFrame(continuous_exposures)

    def single_axis_ppf(
        self,
        axis: str,
        propensity: pd.Series,
        categorical_propensity: pd.Series | None = None,
        categorical_exposure: pd.Series | None = None,
    ) -> pd.Series:
        """Calculate continuous exposures from propensities for a single axis.

        Takes an axis (either 'birth_weight' or 'gestational_age'), a propensity
        and either a categorical propensity or a categorical exposure and
        returns continuous exposures for that axis.

        If categorical propensity is provided rather than exposure, this
        function requires access to the low birth weight and short gestation
        categorical exposure parameters pipeline
        ("risk_factor.low_birth_weight_and_short_gestation.exposure_parameters").

        Parameters
        ----------
        axis
            The axis for which to calculate continuous exposures ('birth_weight'
            or 'gestational_age').
        propensity
            The propensity for the axis.
        categorical_propensity
            The categorical propensity for the axis.
        categorical_exposure
            The categorical exposure for the axis.

        Returns
        -------
            The continuous exposures for the axis.

        Raises
        ------
        ValueError
            If neither categorical propensity nor categorical exposure is provided
            or both are provided.
        """

        if (categorical_propensity is None) == (categorical_exposure is None):
            raise ValueError(
                "Exactly one of categorical propensity or categorical exposure "
                "must be provided."
            )

        if categorical_exposure is None:
            categorical_exposure = super().ppf(categorical_propensity)

        exposure_intervals = categorical_exposure.apply(
            lambda category: self.category_intervals[axis][category]
        )

        exposure_left = exposure_intervals.apply(lambda interval: interval.left)
        exposure_right = exposure_intervals.apply(lambda interval: interval.right)
        continuous_exposure = propensity * (exposure_right - exposure_left) + exposure_left
        return continuous_exposure

    ##################
    # Helper methods #
    ##################

    @staticmethod
    def _parse_description(description: str) -> tuple[pd.Interval, pd.Interval]:
        """Parses a string corresponding to a low birth weight and short gestation
        category to an Interval.

        An example of a standard description:
        'Neonatal preterm and LBWSG (estimation years) - [0, 24) wks, [0, 500) g'
        An example of an edge case for gestational age:
        'Neonatal preterm and LBWSG (estimation years) - [40, 42+] wks, [2000, 2500) g'
        An example of an edge case of birth weight:
        'Neonatal preterm and LBWSG (estimation years) - [36, 37) wks, [4000, 9999] g'
        """
        lbwsg_values = [float(val) for val in re.findall(r"(\d+)", description)]
        if len(list(lbwsg_values)) != 4:
            raise ValueError(
                f"Could not parse LBWSG description '{description}'. Expected 4 numeric values."
            )
        return (
            pd.Interval(*lbwsg_values[:2], closed="left"),  # Gestational Age
            pd.Interval(*lbwsg_values[2:], closed="left"),  # Birth Weight
        )


class LBWSGRisk(Risk):
    exposure_distributions = {"lbwsg": LBWSGDistribution}

    @staticmethod
    def get_continuous_propensity_name(axis: str) -> str:
        return f"{axis}.continuous_propensity"

    @staticmethod
    def get_exposure_name(axis: str) -> str:
        return f"{axis}.exposure"

    ##############
    # Properties #
    ##############

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        configuration_defaults = super().configuration_defaults
        # Add birth exposure data source
        configuration_defaults[self.name]["data_sources"][
            "birth_exposure"
        ] = f"{self.risk}.birth_exposure"
        configuration_defaults[self.name]["distribution_type"] = "lbwsg"
        return configuration_defaults

    @property
    def columns_created(self) -> list[str]:
        columns = [self.categorical_propensity_name]
        for axis in AXES:
            columns.append(self.get_exposure_name(axis))
            columns.append(self.get_continuous_propensity_name(axis))
        return columns

    @property
    def initialization_requirements(self) -> list[str | Resource]:
        return [self.randomness, self.birth_exposure_pipeline]

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self):
        super().__init__("risk_factor.low_birth_weight_and_short_gestation")
        self.categorical_propensity_name = f"{self.risk.name}.categorical_propensity"
        self.birth_exposure_pipeline = f"{self.risk.name}.birth_exposure"

    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self.register_birth_exposure_pipeline(builder)
        self.configuration_age_end = builder.configuration.population.initialization_age_max

    #################
    # Setup methods #
    #################

    def register_exposure_pipeline(self, builder: Builder) -> None:
        builder.value.register_attribute_producer(
            self.exposure_name,
            source=self._get_exposure_source,
            component=self,
            # TODO - MIC-6703: once this is done, we won't needs to specify the required resources here
            required_resources=[self.get_exposure_name(axis) for axis in AXES],
        )

    def register_birth_exposure_pipeline(self, builder: Builder) -> None:
        builder.value.register_attribute_producer(
            self.birth_exposure_pipeline,
            source=[self.exposure_distribution.ppf_pipeline],
            component=self,
            preferred_post_processor=get_exposure_post_processor(builder, self.name),
        )

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        self.initialize_propensities(pop_data)

        if pop_data.user_data.get("age_end", self.configuration_age_end) == 0:
            self.exposure_distribution.exposure_key = "birth_exposure"
        else:
            self.exposure_distribution.exposure_key = "exposure"

        birth_exposures = self.population_view.get_attribute_frame(
            pop_data.index, self.birth_exposure_pipeline
        )
        # Rename the columns
        col_mapping = {axis: self.get_exposure_name(axis) for axis in AXES}
        birth_exposures.rename(columns=col_mapping, inplace=True)
        self.population_view.update(birth_exposures)

    def initialize_propensities(self, pop_data: SimulantData) -> None:
        propensities = {}
        propensities[self.categorical_propensity_name] = self.randomness.get_draw(
            pop_data.index, additional_key=CATEGORICAL
        )
        for axis in AXES:
            propensities[
                self.get_continuous_propensity_name(axis)
            ] = self.randomness.get_draw(pop_data.index, additional_key=axis)

        propensities_df = pd.DataFrame(propensities)
        self.population_view.update(propensities_df)

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def _get_exposure_source(self, index: pd.Index[int]) -> pd.DataFrame:
        exposure_df = self.population_view.get_attributes(
            index, [self.get_exposure_name(axis) for axis in AXES]
        )
        col_mapping = {self.get_exposure_name(axis): axis for axis in AXES}
        return exposure_df.rename(columns=col_mapping)


class LBWSGRiskEffect(RiskEffect):
    TMREL_BIRTH_WEIGHT_INTERVAL: pd.Interval = pd.Interval(3500.0, 4500.0)
    TMREL_GESTATIONAL_AGE_INTERVAL: pd.Interval = pd.Interval(38.0, 42.0)

    ##############
    # Properties #
    ##############

    @property
    def columns_created(self) -> list[str]:
        return self.rr_column_names

    @property
    def initialization_requirements(self) -> list[str | Resource]:
        return ["sex", self.exposure_name]

    @property
    def rr_column_names(self) -> list[str]:
        return [
            self.get_relative_risk_column_name(age_group) for age_group in self.age_intervals
        ]

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, target: str):
        super().__init__("risk_factor.low_birth_weight_and_short_gestation", target)

    def get_relative_risk_column_name(self, age_group_id: str) -> str:
        return (
            f"effect_of_{self.risk.name}_on_{age_group_id}_{self.target.name}_relative_risk"
        )

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.age_intervals = self.get_age_intervals(builder)

        super().setup(builder)
        self.interpolator = self.get_interpolator(builder)

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def build_all_lookup_tables(self, builder: Builder) -> None:
        paf_data, paf_value_cols = self.get_population_attributable_fraction_source(builder)
        self.lookup_tables["population_attributable_fraction"] = self.build_lookup_table(
            builder, paf_data, paf_value_cols
        )

    def get_population_attributable_fraction_source(
        self, builder: Builder
    ) -> tuple[pd.DataFrame, list[str]]:
        paf_key = f"{self.risk}.population_attributable_fraction"
        paf_data = builder.data.load(paf_key)
        return paf_data, builder.data.value_columns()(paf_key)

    def register_target_modifier(self, builder: Builder) -> None:
        builder.value.register_attribute_modifier(
            self.target_name,
            modifier=self.adjust_target,
            component=self,
            required_resources=[self.relative_risk_name],
        )

    def get_age_intervals(self, builder: Builder) -> dict[str, pd.Interval]:
        age_bins = builder.data.load("population.age_bins").set_index("age_start")
        relative_risks = builder.data.load(f"{self.risk}.relative_risk")
        exposed_age_group_starts = (
            relative_risks.groupby("age_start")["value"].any().reset_index()["age_start"]
        )

        return {
            to_snake_case(age_bins.loc[age_start, "age_group_name"]): pd.Interval(
                age_start, age_bins.loc[age_start, "age_end"]
            )
            for age_start in exposed_age_group_starts
        }

    def register_relative_risk_pipeline(self, builder: Builder) -> None:
        builder.value.register_attribute_producer(
            self.relative_risk_name,
            source=self._relative_risk_source,
            component=self,
            required_resources=["age"] + self.rr_column_names,
        )

    def get_interpolator(self, builder: Builder) -> pd.Series:
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

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        pop = self.population_view.get_attributes(pop_data.index, ["sex", self.exposure_name])
        birth_weight = pop[self.exposure_name][BIRTH_WEIGHT]
        gestational_age = pop[self.exposure_name][GESTATIONAL_AGE]

        is_male = pop["sex"] == "Male"
        is_tmrel = (self.TMREL_GESTATIONAL_AGE_INTERVAL.left <= gestational_age) & (
            self.TMREL_BIRTH_WEIGHT_INTERVAL.left <= birth_weight
        )

        def get_relative_risk_for_age_group(age_group: str) -> pd.Series:
            column_name = self.get_relative_risk_column_name(age_group)
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
        pop = self.population_view.get_attributes(index, self.rr_column_names + ["age"])
        relative_risk = pd.Series(1.0, index=index, name=self.relative_risk_name)

        for age_group, interval in self.age_intervals.items():
            age_group_mask = (interval.left <= pop["age"]) & (pop["age"] < interval.right)
            relative_risk[age_group_mask] = pop.loc[
                age_group_mask, self.get_relative_risk_column_name(age_group)
            ]
        return relative_risk

    def get_relative_risk_source(self, builder: Builder) -> Callable[[pd.Index], pd.Series]:
        return self._get_relative_risk
