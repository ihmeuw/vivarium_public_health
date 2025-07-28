"""
###################
# Exposure Effect #
###################

This module contains tools for modeling the relationship between risk
exposure models and disease models.

"""
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from importlib import import_module
from typing import Any

import numpy as np
import pandas as pd
import scipy
from layered_config_tree import ConfigurationError
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.values import Pipeline

from vivarium_public_health.risks.data_transformations import (
    load_exposure_data,
    pivot_categorical,
)
from vivarium_public_health.utilities import EntityString, TargetString, get_lookup_columns

from .exposure import Exposure


class ExposureEffect(Component, ABC):
    """A component to model the effect of a risk-like factor on an affected target.

    This component can source data either from builder.data or from parameters
    supplied in the configuration.

    """

    def __init__(self, entity: str, target: str):
        """

        Parameters
        ----------
        entity
            Type and name of exposure, supplied in the form
            "entity_type.entity_name" where entity_type should be singular (e.g.,
            exposure instead of exposures).
        target
            Type, name, and target rate of entity to be affected by risk factor,
            supplied in the form "entity_type.entity_name.measure"
            where entity_type should be singular (e.g., cause instead of causes).
        """
        super().__init__()
        self.entity = EntityString(entity)
        self.target = TargetString(target)

        self._exposure_distribution_type = None
        self.target_pipeline_name = f"{self.target.name}.{self.target.measure}"
        self.target_paf_pipeline_name = f"{self.target_pipeline_name}.paf"

    ###############
    # Properties #
    ##############

    @property
    def name(self) -> str:
        return self.get_name(self.entity, self.target)

    @staticmethod
    @abstractmethod
    def get_name(self) -> Callable[[EntityString, TargetString], str]:
        """Abstract property that must be implemented by subclasses to provide a naming function."""
        raise NotImplementedError

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        """Default values for any configurations managed by this component."""
        return {
            self.name: {
                "data_sources": {
                    "relative_risk": f"{self.entity}.relative_risk",
                    "population_attributable_fraction": f"{self.entity}.population_attributable_fraction",
                },
                "data_source_parameters": {
                    "relative_risk": {},
                },
            }
        }

    @property
    def is_exposure_categorical(self) -> bool:
        return self._exposure_distribution_type in [
            "dichotomous",
            "ordered_polytomous",
            "unordered_polytomous",
        ]

    #####################
    # Lifecycle methods #
    #####################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.measure = self.get_exposure_callable(builder)

        self._relative_risk_source = self.get_relative_risk_source(builder)
        self.relative_risk = self.get_relative_risk_pipeline(builder)

        self.register_target_modifier(builder)
        self.register_paf_modifier(builder)

    #################
    # Setup methods #
    #################

    def setup_component(self, builder: Builder) -> None:
        self.exposure_component = self._get_exposure_class(builder)
        self.exposure_pipeline_name = (
            f"{self.entity.name}.{self.exposure_component.exposure_type}"
        )
        super().setup_component(builder)

    def build_all_lookup_tables(self, builder: Builder) -> None:
        self._exposure_distribution_type = self.get_distribution_type(builder)

        rr_data = self.load_relative_risk(builder)
        rr_value_cols = None
        if self.is_exposure_categorical:
            rr_data, rr_value_cols = self.process_categorical_data(builder, rr_data)
        self.lookup_tables["relative_risk"] = self.build_lookup_table(
            builder, rr_data, rr_value_cols
        )

        paf_data = self.get_filtered_data(
            builder, self.configuration.data_sources.population_attributable_fraction
        )
        self.lookup_tables["population_attributable_fraction"] = self.build_lookup_table(
            builder, paf_data
        )

    def get_distribution_type(self, builder: Builder) -> str:
        """Get the distribution type for the risk from the configuration."""
        if self.exposure_component.distribution_type:
            return self.exposure_component.distribution_type
        return self.exposure_component.get_distribution_type(builder)

    def load_relative_risk(
        self,
        builder: Builder,
        configuration=None,
    ) -> str | float | pd.DataFrame:
        if configuration is None:
            configuration = self.configuration

        rr_source = configuration.data_sources.relative_risk
        rr_dist_parameters = configuration.data_source_parameters.relative_risk.to_dict()

        if isinstance(rr_source, str):
            try:
                distribution = getattr(import_module("scipy.stats"), rr_source)
                rng = np.random.default_rng(builder.randomness.get_seed(self.name))
                rr_data = distribution(**rr_dist_parameters).ppf(rng.random())
            except AttributeError:
                rr_data = self.get_filtered_data(builder, rr_source)
            except TypeError:
                raise ConfigurationError(
                    f"Parameters {rr_dist_parameters} are not valid for distribution {rr_source}."
                )
        else:
            rr_data = self.get_filtered_data(builder, rr_source)
        return rr_data

    def get_filtered_data(
        self, builder: "Builder", data_source: str | float | pd.DataFrame
    ) -> float | pd.DataFrame:
        data = super().get_data(builder, data_source)

        if isinstance(data, pd.DataFrame):
            # filter data to only include the target entity and measure
            correct_target_mask = True
            columns_to_drop = []
            if "affected_entity" in data.columns:
                correct_target_mask &= data["affected_entity"] == self.target.name
                columns_to_drop.append("affected_entity")
            if "affected_measure" in data.columns:
                correct_target_mask &= data["affected_measure"] == self.target.measure
                columns_to_drop.append("affected_measure")
            data = data[correct_target_mask].drop(columns=columns_to_drop)
        return data

    def process_categorical_data(
        self, builder: Builder, rr_data: str | float | pd.DataFrame
    ) -> tuple[str | float | pd.DataFrame, list[str]]:
        if not isinstance(rr_data, pd.DataFrame):
            exposed = builder.data.load("population.demographic_dimensions")
            exposed[
                "parameter"
            ] = self.exposure_component.dichotomous_exposure_category_names.exposed
            exposed["value"] = rr_data
            unexposed = exposed.copy()
            unexposed[
                "parameter"
            ] = self.exposure_component.dichotomous_exposure_category_names.unexposed
            unexposed["value"] = 1
            rr_data = pd.concat([exposed, unexposed], ignore_index=True)
        if "parameter" in rr_data.index.names:
            rr_data = rr_data.reset_index("parameter")

        rr_value_cols = list(rr_data["parameter"].unique())
        rr_data = pivot_categorical(builder, self.entity, rr_data, "parameter")
        return rr_data, rr_value_cols

    # todo currently this isn't being called. we need to properly set rrs if
    #  the exposure has been rebinned
    def rebin_relative_risk_data(
        self, builder, relative_risk_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Rebin relative risk data.

        When the polytomous risk is rebinned, matching relative risk needs to be rebinned.
        After rebinning, rr for both exposed and unexposed categories should be the weighted sum of relative risk
        of the component categories where weights are relative proportions of exposure of those categories.
        For example, if cat1, cat2, cat3 are exposed categories and cat4 is unexposed with exposure [0.1,0.2,0.3,0.4],
        for the matching rr = [rr1, rr2, rr3, 1], rebinned rr for the rebinned cat1 should be:
        (0.1 *rr1 + 0.2 * rr2 + 0.3* rr3) / (0.1+0.2+0.3)
        """
        if not self.entity in builder.configuration.to_dict():
            return relative_risk_data

        rebin_exposed_categories = set(builder.configuration[self.entity]["rebinned_exposed"])

        if rebin_exposed_categories:
            # todo make sure this works
            exposure_data = load_exposure_data(builder, self.entity)
            relative_risk_data = self._rebin_relative_risk_data(
                relative_risk_data, exposure_data, rebin_exposed_categories
            )

        return relative_risk_data

    def _rebin_relative_risk_data(
        self,
        relative_risk_data: pd.DataFrame,
        exposure_data: pd.DataFrame,
        rebin_exposed_categories: set,
    ) -> pd.DataFrame:
        cols = list(exposure_data.columns.difference(["value"]))

        relative_risk_data = relative_risk_data.merge(exposure_data, on=cols)
        relative_risk_data["value_x"] = relative_risk_data.value_x.multiply(
            relative_risk_data.value_y
        )
        relative_risk_data.parameter = relative_risk_data["parameter"].map(
            lambda p: "cat1" if p in rebin_exposed_categories else "cat2"
        )
        relative_risk_data = relative_risk_data.groupby(cols).sum().reset_index()
        relative_risk_data["value"] = relative_risk_data.value_x.divide(
            relative_risk_data.value_y
        ).fillna(0)
        return relative_risk_data.drop(columns=["value_x", "value_y"])

    def get_exposure_callable(self, builder: Builder) -> Callable[[pd.Index], pd.Series]:
        return builder.value.get_value(self.exposure_pipeline_name)

    def adjust_target(self, index: pd.Index, target: pd.Series) -> pd.Series:
        relative_risk = self.relative_risk(index)
        return target * relative_risk

    def get_relative_risk_source(self, builder: Builder) -> Callable[[pd.Index], pd.Series]:

        if not self.is_exposure_categorical:
            tmred = builder.data.load(f"{self.entity}.tmred")
            tmrel = 0.5 * (tmred["min"] + tmred["max"])
            scale = builder.data.load(f"{self.entity}.relative_risk_scalar")

            def generate_relative_risk(index: pd.Index) -> pd.Series:
                rr = self.lookup_tables["relative_risk"](index)
                exposure = self.measure(index)
                relative_risk = np.maximum(rr.values ** ((exposure - tmrel) / scale), 1)
                return relative_risk

        else:
            index_columns = ["index", self.entity.name]

            def generate_relative_risk(index: pd.Index) -> pd.Series:
                rr = self.lookup_tables["relative_risk"](index)
                exposure = self.measure(index).reset_index()
                exposure.columns = index_columns
                exposure = exposure.set_index(index_columns)

                relative_risk = rr.stack().reset_index()
                relative_risk.columns = index_columns + ["value"]
                # Check if we need to remap cat1 and cat2 to exposed and unexposed categories
                if (
                    "cat1" in relative_risk[self.entity.name].unique()
                    and self._exposure_distribution_type == "dichotomous"
                ):
                    warnings.warn(
                        "Using 'cat1' and 'cat2' for dichotomous exposure is deprecated and will be removed in a future release. Use 'exposed' and 'unexposed' instead.",
                        FutureWarning,
                        stacklevel=2,
                    )
                    relative_risk[self.entity.name] = relative_risk[self.entity.name].replace(
                        {
                            "cat1": self.exposure_component.dichotomous_exposure_category_names.exposed,
                            "cat2": self.exposure_component.dichotomous_exposure_category_names.unexposed,
                        }
                    )
                relative_risk = relative_risk.set_index(index_columns)

                effect = relative_risk.loc[exposure.index, "value"].droplevel(
                    self.entity.name
                )
                return effect

        return generate_relative_risk

    def get_relative_risk_pipeline(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            f"{self.entity.name}_on_{self.target.name}.relative_risk",
            self._relative_risk_source,
            component=self,
            required_resources=[self.measure],
        )

    def register_target_modifier(self, builder: Builder) -> None:
        builder.value.register_value_modifier(
            self.target_pipeline_name,
            modifier=self.adjust_target,
            component=self,
            required_resources=[self.relative_risk],
        )

    def register_paf_modifier(self, builder: Builder) -> None:
        required_columns = get_lookup_columns(
            [self.lookup_tables["population_attributable_fraction"]]
        )
        builder.value.register_value_modifier(
            self.target_paf_pipeline_name,
            modifier=self.lookup_tables["population_attributable_fraction"],
            component=self,
            required_resources=required_columns,
        )

    ##################
    # Helper methods #
    ##################

    def _get_exposure_class(self, builder: Builder) -> Exposure:
        exposure_component = builder.components.get_component(self.entity)
        if not isinstance(exposure_component, Exposure):
            raise ValueError(
                f"Exposure model {self.name} requires an Exposure component named {self.entity}"
            )
        return exposure_component
