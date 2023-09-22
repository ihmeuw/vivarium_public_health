"""
=====================
Linear Scale-Up Model
=====================

This module contains tools for applying a linear scale-up to an intervention

"""
from datetime import datetime
from typing import Any, Callable, Dict, Tuple

import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.lookup import LookupTable
from vivarium.framework.time import Time, get_time_stamp
from vivarium.framework.values import Pipeline

from vivarium_public_health.utilities import EntityString


class LinearScaleUp(Component):
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

    CONFIGURATION_DEFAULTS = {
        "treatment": {
            "date": {
                "start": "start",
                "end": "end",
            },
            "value": {
                "start": "data",
                "end": "data",
            },
        }
    }

    ##############
    # Properties #
    ##############

    @property
    def configuration_defaults(self) -> Dict[str, Any]:
        return {self.configuration_key: self.CONFIGURATION_DEFAULTS["treatment"]}

    @property
    def configuration_key(self) -> str:
        return f"{self.treatment.name}_scale_up"

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, treatment: str):
        """
        Parameters
        ----------
        treatment :
            the type and name of a treatment, specified as "type.name". Type is singular.
        """
        super().__init__()
        self.treatment = EntityString(treatment)

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        """Perform this component's setup."""
        self.is_intervention_scenario = self.get_is_intervention_scenario(builder)
        self.clock = self.get_clock(builder)
        self.scale_up_start_date, self.scale_up_end_date = self.get_scale_up_dates(builder)
        self.scale_up_start_value, self.scale_up_end_value = self.get_scale_up_values(builder)

        self.pipelines = self.get_required_pipelines(builder)

        self.register_intervention_modifiers(builder)

    # noinspection PyMethodMayBeStatic
    def get_is_intervention_scenario(self, builder: Builder) -> bool:
        return builder.configuration.intervention.scenario != "baseline"

    # noinspection PyMethodMayBeStatic
    def get_clock(self, builder: Builder) -> Callable[[], Time]:
        return builder.time.clock()

    # noinspection PyMethodMayBeStatic
    def get_scale_up_dates(self, builder: Builder) -> Tuple[datetime, datetime]:
        scale_up_config = builder.configuration[self.configuration_key]["date"]

        def get_endpoint(endpoint_type: str) -> datetime:
            if scale_up_config[endpoint_type] == endpoint_type:
                endpoint = get_time_stamp(builder.configuration.time[endpoint_type])
            else:
                endpoint = get_time_stamp(scale_up_config[endpoint_type])
            return endpoint

        return get_endpoint("start"), get_endpoint("end")

    def get_scale_up_values(self, builder: Builder) -> Tuple[LookupTable, LookupTable]:
        scale_up_config = builder.configuration[self.configuration_key]["value"]

        def get_endpoint_value(endpoint_type: str) -> LookupTable:
            if scale_up_config[endpoint_type] == "data":
                endpoint = self.get_endpoint_value_from_data(builder, endpoint_type)
            else:
                endpoint = builder.lookup.build_table(scale_up_config[endpoint_type])
            return endpoint

        return get_endpoint_value("start"), get_endpoint_value("end")

    # noinspection PyMethodMayBeStatic
    def get_required_pipelines(self, builder: Builder) -> Dict[str, Pipeline]:
        return {}

    def register_intervention_modifiers(self, builder: Builder):
        builder.value.register_value_modifier(
            f"{self.treatment}.exposure_parameters",
            modifier=self.coverage_effect,
        )

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def coverage_effect(self, idx: pd.Index, target: pd.Series) -> pd.Series:
        if not self.is_intervention_scenario or self.clock() < self.scale_up_start_date:
            progress = 0.0
        elif self.scale_up_start_date <= self.clock() < self.scale_up_end_date:
            progress = (self.clock() - self.scale_up_start_date) / (
                self.scale_up_end_date - self.scale_up_start_date
            )
        else:
            progress = 1.0

        target = self.apply_scale_up(idx, target, progress) if progress else target
        return target

    ##################
    # Helper methods #
    ##################

    def get_endpoint_value_from_data(
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

    def apply_scale_up(
        self, idx: pd.Index, target: pd.Series, scale_up_progress: float
    ) -> pd.Series:
        start_value = self.scale_up_start_value(idx)
        end_value = self.scale_up_end_value(idx)
        value_increase = scale_up_progress * (end_value - start_value)

        target.loc[idx] += value_increase
        return target
