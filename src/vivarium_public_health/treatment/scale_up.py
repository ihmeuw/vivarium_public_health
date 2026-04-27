"""
=====================
Linear Scale-Up Model
=====================

This module contains tools for applying a linear scale-up to an intervention

"""

from collections.abc import Callable
from typing import Any

import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.lookup import LookupTable
from vivarium.framework.values import Pipeline
from vivarium.types import Time

from vivarium_public_health.utilities import EntityString


class LinearScaleUp(Component):
    """Apply a linear scale-up to intervention coverage over a configured time period.

    This component linearly interpolates an intervention's exposure parameter
    between a start value and an end value over a specified date range. Before
    the start date, the start value is used; after the end date, the end value
    is used. Endpoint values can be sourced from artifact data or from scalar
    parameters in the configuration.

    For example, for an intervention called ``treatment`` the configuration
    could look like this:

    .. code-block:: yaml

        configuration:
            treatment_scale_up:
                date:
                    start: "2020-01-01"
                    end: "2020-12-31"
                value:
                    start: 0.0
                    end: 0.9

    """

    CONFIGURATION_DEFAULTS = {
        "treatment": {
            "date": {
                "start": "2020-01-01",
                "end": "2020-12-31",
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
    def configuration_defaults(self) -> dict[str, Any]:
        """Provides default configuration values for this component.

        Configuration structure::

            {treatment_name}_scale_up:
                date:
                    start: str
                        Start date for the scale-up period in ISO format
                        (``"YYYY-MM-DD"``). Default is ``"2020-01-01"``.
                    end: str
                        End date for the scale-up period in ISO format.
                        Default is ``"2020-12-31"``.
                value:
                    start: str or float
                        Value at the start of scale-up. Can be ``"data"``
                        to load from artifact, or a numeric value.
                        Default is ``"data"``.
                    end: str or float
                        Value at the end of scale-up. Can be ``"data"``
                        to load from artifact, or a numeric value.
                        Default is ``"data"``.

        The scale-up linearly interpolates between start and end values
        over the specified date range. Outside this range, values are
        clamped to the nearest endpoint value.
        """
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
        treatment
            The type and name of a treatment, specified as
            ``"type.name"``. Type is singular.
        """
        super().__init__()
        self.treatment = EntityString(treatment)

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        """Set up the component by loading dates, values, and registering the modifier.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.
        """
        self.is_intervention_scenario = self.get_is_intervention_scenario(builder)
        self.clock = self.get_clock(builder)
        self.scale_up_start_date, self.scale_up_end_date = self.get_scale_up_dates(builder)
        self.scale_up_start_value, self.scale_up_end_value = self.get_scale_up_values(builder)

        self.pipelines = self.get_required_pipelines(builder)

        self.register_intervention_modifiers(builder)

    # noinspection PyMethodMayBeStatic
    def get_is_intervention_scenario(self, builder: Builder) -> bool:
        """Determine whether the current simulation is an intervention scenario.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.

        Returns
        -------
            ``True`` if the configured intervention scenario is not
            ``"baseline"``.
        """
        return builder.configuration.intervention.scenario != "baseline"

    # noinspection PyMethodMayBeStatic
    def get_clock(self, builder: Builder) -> Callable[[], Time]:
        """Return the simulation clock callable.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.

        Returns
        -------
            A callable that returns the current simulation time.
        """
        return builder.time.clock()

    # noinspection PyMethodMayBeStatic
    def get_scale_up_dates(self, builder: Builder) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Load the scale-up start and end dates from the configuration.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.

        Returns
        -------
            A tuple of ``(start_date, end_date)`` as :class:`pandas.Timestamp`
            objects.
        """
        scale_up_config = builder.configuration[self.configuration_key]["date"]
        return pd.Timestamp(scale_up_config["start"]), pd.Timestamp(scale_up_config["end"])

    def get_scale_up_values(self, builder: Builder) -> tuple[LookupTable, LookupTable]:
        """Get the values at the start and end of the scale-up period.

        Parameters
        ----------
        builder
            Interface to access simulation managers.

        Returns
        -------
            A tuple of lookup tables returning the values at the start and end
            of the scale-up period.
        """
        scale_up_config = builder.configuration[self.configuration_key]["value"]

        def get_endpoint_value(endpoint_type: str) -> LookupTable:
            if scale_up_config[endpoint_type] == "data":
                endpoint = self.get_endpoint_value_from_data(builder, endpoint_type)
            else:
                endpoint = self.build_lookup_table(
                    builder, "endpoint", scale_up_config[endpoint_type]
                )
            return endpoint

        return get_endpoint_value("start"), get_endpoint_value("end")

    # noinspection PyMethodMayBeStatic
    def get_required_pipelines(self, builder: Builder) -> dict[str, Pipeline]:
        """Return any additional pipelines required by subclasses.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.

        Returns
        -------
            A dictionary mapping pipeline names to
            :class:`~vivarium.framework.values.pipeline.Pipeline` objects.
            The base implementation returns an empty dictionary.
        """
        return {}

    def register_intervention_modifiers(self, builder: Builder) -> None:
        """Register the coverage effect modifier on the treatment's exposure pipeline.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.
        """
        builder.value.register_attribute_modifier(
            f"{self.treatment}.exposure_parameters", modifier=self.coverage_effect
        )

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def coverage_effect(self, idx: pd.Index, target: pd.Series) -> pd.Series:
        """Modify the treatment's exposure parameters based on the current scale-up progress.

        Compute the linear interpolation progress between the scale-up start
        and end dates, then apply :meth:`apply_scale_up` to adjust the target
        values.

        Parameters
        ----------
        idx
            Index of the simulants to modify.
        target
            Current exposure parameter values for the given simulants.

        Returns
        -------
            The modified exposure parameter values.
        """
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
        """Get the value at the start or end of the scale-up period from data.

        Parameters
        ----------
        builder
            Interface to access simulation managers.
        endpoint_type
            The type of endpoint to get the value for. Allowed values are
            "start" and "end".

        Returns
        -------
            A lookup table returning the value at the start or end of the
            scale-up period.
        """
        if endpoint_type == "start":
            endpoint_data = builder.data.load(f"{self.treatment}.exposure")
        elif endpoint_type == "end":
            endpoint_data = builder.data.load(f"alternate_{self.treatment}.exposure")
        else:
            raise ValueError(
                f'Invalid endpoint type {endpoint_type}. Allowed types are "start" and "end".'
            )
        return self.build_lookup_table(builder, "endpoint", endpoint_data)

    def apply_scale_up(
        self, idx: pd.Index, target: pd.Series, scale_up_progress: float
    ) -> pd.Series:
        """Apply the linearly interpolated scale-up adjustment to the target values.

        The adjustment is computed as:

            adjustment = progress * (end_value - start_value)

        and is added to the current target values.

        Parameters
        ----------
        idx
            Index of the simulants to modify.
        target
            Current target values.
        scale_up_progress
            A float between 0.0 and 1.0 representing how far through
            the scale-up period the simulation has progressed.

        Returns
        -------
            The target values with the scale-up adjustment applied.
        """
        start_value = self.scale_up_start_value(idx)
        end_value = self.scale_up_end_value(idx)
        value_increase = scale_up_progress * (end_value - start_value)

        target.loc[idx] += value_increase
        return target
