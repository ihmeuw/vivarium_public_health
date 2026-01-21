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
from vivarium.framework.lookup import LookupTable
from vivarium.framework.population import SimulantData

from vivarium_public_health.risks import Risk, RiskEffect
from vivarium_public_health.risks.data_transformations import (
    get_exposure_post_processor,
    pivot_categorical,
)
from vivarium_public_health.risks.distributions import PolytomousDistribution
from vivarium_public_health.utilities import EntityString, to_snake_case

CATEGORICAL = "categorical"
BIRTH_WEIGHT = "birth_weight"
GESTATIONAL_AGE = "gestational_age"
AXES = [BIRTH_WEIGHT, GESTATIONAL_AGE]


class LBWSGDistribution(PolytomousDistribution):
    @property
    def categories(self) -> list[str]:
        # These need to be sorted so the cumulative sum is in the correct order of categories
        # and results are therefore reproducible and correct
        lookup_table = (
            self.exposure_params_table
            if self.exposure_data_type == "exposure"
            else self.birth_exposure_params_table
        )
        return sorted(lookup_table.value_columns)

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
        self.exposure_data_type = "birth_exposure"
        self.birth_exposure_ppf_pipeline = f"{self.risk}.birth_exposure_ppf"
        self.birth_exposure_params_pipeline = f"{self.risk}.birth_exposure_parameters"
        self.risk_propensity = f"{self.risk.name}.categorical_propensity"

    def setup(self, builder: Builder) -> None:
        self.birth_exposure_params_table = self.build_birth_exposure_params_table(builder)
        super().setup(builder)
        self.category_intervals = self.get_category_intervals(builder)

        if self.birth_exposure_params_table is None and self.exposure_params_table is None:
            raise ConfigurationError(
                "The LBWSG distribution requires either 'birth_exposure' or 'exposure' data"
                " to be available in the simulation."
            )

    def register_exposure_ppf_pipeline(self, builder):
        required_resources = [self.exposure_params_pipeline, self.risk_propensity] + [
            LBWSGRisk.get_continuous_propensity_name(axis) for axis in AXES
        ]
        builder.value.register_attribute_producer(
            self.exposure_ppf_pipeline,
            source=self.exposure_ppf,
            required_resources=required_resources,
        )

    def register_exposure_params_pipeline(self, builder: Builder) -> None:
        lookup_tables = [
            table
            for table in [self.exposure_params_table, self.birth_exposure_params_table]
            if table is not None
        ]

        builder.value.register_attribute_producer(
            self.exposure_params_pipeline,
            source=self.get_exposure_parameters,
            required_resources=lookup_tables,
        )

    def build_exposure_params_table(self, builder: Builder) -> LookupTable | None:
        try:
            return super().build_exposure_params_table(builder)
        except ConfigurationError:
            logger.warning(
                "The data for LBWSG exposure is missing from the simulation. LBWSG will not"
                " be able to initialize neonatal simulants."
            )

    def build_birth_exposure_params_table(self, builder: Builder) -> LookupTable | None:
        try:
            data = self.get_data(
                builder, self.configuration["data_sources"]["birth_exposure"]
            )
            value_columns = self.get_exposure_value_columns(data)
            if isinstance(data, pd.DataFrame):
                data = pivot_categorical(data, "parameter")

            return self.build_lookup_table(
                builder, "birth_exposure", data_source=data, value_columns=value_columns
            )
        except ConfigurationError:
            logger.warning(
                "Birth exposure data for LBWSG is missing from the simulation. LBWSG will"
                " not be able to initialize newborn simulants."
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

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def exposure_ppf(self, index: pd.Index) -> pd.DataFrame:
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

        categorical_exposure = super().exposure_ppf(index=propensities.index)
        continuous_exposures = {
            axis: self._single_axis_ppf(
                axis,
                propensities[LBWSGRisk.get_continuous_propensity_name(axis)],
                categorical_exposure=categorical_exposure,
            )
            for axis in AXES
        }
        return pd.DataFrame(continuous_exposures)

    def get_exposure_parameters(self, index: pd.Index) -> pd.DataFrame:
        if self.exposure_data_type == "exposure":
            if self.exposure_params_table is None:
                raise ConfigurationError(
                    "LBWSG exposure data is missing from the simulation. Cannot initialize"
                    " neonatal simulants."
                )
            return self.exposure_params_table(index)
        else:
            if self.birth_exposure_params_table is None:
                raise ConfigurationError(
                    "LBWSG birth exposure data is missing from the simulation. Cannot"
                    " initialize newborn simulants."
                )
            return self.birth_exposure_params_table(index)

    ##################
    # Helper methods #
    ##################

    def _single_axis_ppf(
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
            categorical_exposure = super().exposure_ppf(categorical_propensity)

        exposure_intervals = categorical_exposure.apply(
            lambda category: self.category_intervals[axis][category]
        )

        exposure_left = exposure_intervals.apply(lambda interval: interval.left)
        exposure_right = exposure_intervals.apply(lambda interval: interval.right)
        continuous_exposure = propensity * (exposure_right - exposure_left) + exposure_left
        return continuous_exposure

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
        """Provides default configuration values for this component.

        Extends the base Risk configuration with LBWSG-specific settings.

        Configuration structure::

            {risk_name}:
                data_sources:
                    exposure:
                        Source for exposure data. Inherited from Risk.
                    ensemble_distribution_weights:
                        Source for ensemble weights. Inherited from Risk.
                    exposure_standard_deviation:
                        Source for exposure SD. Inherited from Risk.
                    birth_exposure:
                        Source for birth exposure data specific to LBWSG.
                        Default is the artifact key
                        ``{risk}.birth_exposure``. This provides the
                        joint distribution of birth weight and gestational
                        age categories at birth.
                distribution_type: str
                    Fixed to ``"lbwsg"`` for this component, using the
                    specialized LBWSGDistribution.
        """
        configuration_defaults = super().configuration_defaults
        # Add birth exposure data source
        configuration_defaults[self.name]["data_sources"][
            "birth_exposure"
        ] = f"{self.risk}.birth_exposure"
        configuration_defaults[self.name]["distribution_type"] = "lbwsg"
        return configuration_defaults

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

        continuous_propensity_columns = [
            self.get_continuous_propensity_name(axis) for axis in AXES
        ]
        builder.population.register_initializer(
            initializer=self.initialize_propensities,
            columns=[self.categorical_propensity_name, *continuous_propensity_columns],
            required_resources=[self.randomness],
        )

        builder.population.register_initializer(
            initializer=self.initialize_exposures,
            columns=[self.get_exposure_name(axis) for axis in AXES],
            required_resources=[self.birth_exposure_pipeline],
        )

    #################
    # Setup methods #
    #################

    def register_exposure_pipeline(self, builder: Builder) -> None:
        builder.value.register_attribute_producer(
            self.exposure_name,
            source=self._get_exposure_source,
            # TODO - MIC-6703: once this is done, we won't needs to specify the required resources here
            required_resources=[self.get_exposure_name(axis) for axis in AXES],
        )

    def register_birth_exposure_pipeline(self, builder: Builder) -> None:
        builder.value.register_attribute_producer(
            self.birth_exposure_pipeline,
            source=[self.exposure_distribution.exposure_ppf_pipeline],
            preferred_post_processor=get_exposure_post_processor(builder, self.name),
        )

    ########################
    # Event-driven methods #
    ########################

    def initialize_exposures(self, pop_data: SimulantData) -> None:

        if pop_data.user_data.get("age_end", self.configuration_age_end) == 0:
            self.exposure_distribution.exposure_data_type = "birth_exposure"
        else:
            self.exposure_distribution.exposure_data_type = "exposure"

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

    def setup(self, builder: Builder) -> None:
        self.age_intervals = self.get_age_intervals(builder)
        super().setup(builder)
        self.interpolator = self.get_interpolator(builder)
        builder.population.register_initializer(
            initializer=self.on_initialize_simulants,
            columns=self.rr_column_names,
            required_resources=[self.exposure_name, "sex"],
        )

    #################
    # Setup methods #
    #################

    def build_rr_lookup_table(self, builder: Builder) -> None:
        # We don't need a LookupTable for RR since we are using interpolators
        pass

    def build_paf_lookup_table(self, builder):
        return self.build_lookup_table(builder, "population_attributable_fraction")

    def register_target_modifier(self, builder: Builder) -> None:
        builder.value.register_attribute_modifier(
            self.target_name,
            modifier=self.adjust_target,
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
