"""
=====================
Linear Scale-Up Model
=====================

This module contains tools for applying a linear scale-up to an intervention

"""
from datetime import datetime
from typing import Callable, Dict, List, Tuple

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.lookup import LookupTable
from vivarium.framework.population import PopulationView
from vivarium.framework.time import Time, get_time_stamp
from vivarium.framework.values import Pipeline

from vivarium_public_health.utilities import EntityString


class LinearScaleUp:
    """
    A model for applying a linear scale-up to an intervention.

    This component requires input data for beginning and end dates, as well as
    beginning and end values. Scale-up start and end dates are by default the
    beginning and end of the simulation, but they can both be set to other
    values in the configuration. Data for the values at scale-up endpoints can
    be sourced either from builder.data or from parameters provided in the
    configuration. For example, for an intervention called 'treatment' the
    configuration could look like this:

    .. code-block:: yaml

        configuration:
            treatment_scale_up:
                start:
                    date:
                        year: 2020
                        month: 1
                        day: 1
                    value: 0.0
                end:
                    date:
                        year: 2020
                        month: 12
                        day: 31
                    value: 0.9

    """

    configuration_defaults = {
        "treatment": {
            "start": {
                "date": "start",
                "value": "data",
            },
            "end": {
                "date": "end",
                "value": "data",
            },
        }
    }

    def __init__(self, treatment: str):
        """
        Parameters
        ----------
        treatment :
            the type and name of a treatment, specified as "type.name". Type is singular.
        """
        self.treatment = EntityString(treatment)
        self.configuration_defaults = self._get_configuration_defaults()

    ##########################
    # Initialization methods #
    ##########################

    def _get_configuration_defaults(self) -> Dict[str, Dict]:
        return {self.configuration_key: LinearScaleUp.configuration_defaults["treatment"]}

    ##############
    # Properties #
    ##############

    @property
    def name(self) -> str:
        return f"{self.treatment.name}_intervention"

    @property
    def configuration_key(self) -> str:
        return f"{self.treatment.name}_scale_up"

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        """Perform this component's setup."""
        self.is_intervention_scenario = self._get_is_intervention_scenario(builder)
        self.clock = self._get_clock(builder)
        self.scale_up_start_date, self.scale_up_end_date = self._get_scale_up_date_endpoints(
            builder
        )
        (
            self.scale_up_start_value,
            self.scale_up_end_value,
        ) = self._get_scale_up_value_endpoints(builder)

        required_columns = self._get_required_columns()
        self.pipelines = self._get_required_pipelines(builder)

        self._register_intervention_modifiers(builder)

        if required_columns:
            self.population_view = self._get_population_view(builder, required_columns)

    # noinspection PyMethodMayBeStatic
    def _get_is_intervention_scenario(self, builder: Builder) -> bool:
        return builder.configuration.intervention.scenario != "baseline"

    # noinspection PyMethodMayBeStatic
    def _get_clock(self, builder: Builder) -> Callable[[], Time]:
        return builder.time.clock()

    # noinspection PyMethodMayBeStatic
    def _get_scale_up_date_endpoints(self, builder: Builder) -> Tuple[datetime, datetime]:
        scale_up_config = builder.configuration[self.configuration_key]

        def get_endpoint(endpoint_type: str) -> datetime:
            if scale_up_config[endpoint_type]["date"] == endpoint_type:
                endpoint = get_time_stamp(builder.configuration.time[endpoint_type])
            else:
                endpoint = get_time_stamp(scale_up_config[endpoint_type]["date"])
            return endpoint

        return get_endpoint("start"), get_endpoint("end")

    def _get_scale_up_value_endpoints(
        self, builder: Builder
    ) -> Tuple[LookupTable, LookupTable]:
        scale_up_config = builder.configuration[self.configuration_key]

        def get_endpoint_value(endpoint_type: str) -> LookupTable:
            if scale_up_config[endpoint_type]["value"] == "data":
                endpoint = self._get_endpoint_value_from_data(builder, endpoint_type)
            else:
                endpoint = builder.lookup.build_table(scale_up_config[endpoint_type]["value"])
            return endpoint

        return get_endpoint_value("start"), get_endpoint_value("end")

    # noinspection PyMethodMayBeStatic
    def _get_required_columns(self) -> List[str]:
        return []

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def _get_required_pipelines(self, builder: Builder) -> Dict[str, Pipeline]:
        return {}

    def _register_intervention_modifiers(self, builder: Builder):
        builder.value.register_value_modifier(
            f"{self.treatment}.exposure_parameters",
            modifier=self._coverage_effect,
        )

    # noinspection PyMethodMayBeStatic
    def _get_population_view(
        self, builder: Builder, required_columns: List[str]
    ) -> PopulationView:
        return builder.population.get_view(required_columns)

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def _coverage_effect(self, idx: pd.Index, target: pd.Series) -> pd.Series:
        if not self.is_intervention_scenario or self.clock() < self.scale_up_start_date:
            scale_up_progress = 0.0
        elif self.scale_up_start_date <= self.clock() < self.scale_up_end_date:
            scale_up_progress = (self.clock() - self.scale_up_start_date) / (
                self.scale_up_end_date - self.scale_up_start_date
            )
        else:
            scale_up_progress = 1.0

        target = (
            self._apply_scale_up(idx, target, scale_up_progress)
            if scale_up_progress
            else target
        )
        return target

    ##################
    # Helper methods #
    ##################

    def _get_endpoint_value_from_data(
        self, builder: Builder, endpoint_type: str
    ) -> LookupTable:
        if endpoint_type == "start":
            endpoint_data = builder.data.load(f"{self.treatment}.exposure")
        elif endpoint_type == "end":
            endpoint_data = builder.data.load(f"alternate_{self.treatment}.exposure")
        else:
            raise ValueError(
                f'Invalid endpoint type {endpoint_type}. Allowed types are "start" and "end".'
            )
        return builder.lookup.build_table(endpoint_data)

    def _apply_scale_up(
        self, idx: pd.Index, target: pd.Series, scale_up_progress: float
    ) -> pd.Series:
        start_value = self.scale_up_start_value(idx)
        end_value = self.scale_up_end_value(idx)
        value_increase = scale_up_progress * (end_value - start_value)

        target.loc[idx] += value_increase
        return target
