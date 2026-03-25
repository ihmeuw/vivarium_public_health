"""
==========================
Simple Intervention Models
==========================

This module contains simple intervention models that work at the population
level by providing direct shifts to epidemiological measures.

"""

from __future__ import annotations

from typing import Any

import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder

from vivarium_public_health.utilities import TargetString


class AbsoluteShift(Component):
    """Apply an absolute shift to a target epidemiological measure.

    This component registers a value modifier on the target pipeline that
    replaces the current value with a configured absolute value for
    simulants within a specified age range. When the configured
    ``target_value`` is ``"baseline"``, no modification is applied.

    The target measure is specified at instantiation as a
    :class:`~vivarium_public_health.utilities.TargetString` of the form
    ``"entity.measure"`` (e.g. ``"cause.incidence_rate"``).

    """

    CONFIGURATION_DEFAULTS = {
        "intervention": {
            "target_value": "baseline",
            "age_start": 0,
            "age_end": 125,
        }
    }

    ##############
    # Properties #
    ##############

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        """Provides default configuration values for this intervention.

        Configuration structure::

            intervention_on_{target_name}:
                target_value: str or float
                    Value to set for the target measure. Use ``"baseline"``
                    to apply no intervention effect, or a numeric value to
                    set an absolute value for the measure. Default is
                    ``"baseline"`` (no effect).
                age_start: float
                    Minimum age (in years) for the intervention to apply.
                    Simulants below this age are unaffected. Default is 0.
                age_end: float
                    Maximum age (in years) for the intervention to apply.
                    Simulants above this age are unaffected. Default is 125.
        """
        return {
            f"intervention_on_{self.target.name}": self.CONFIGURATION_DEFAULTS["intervention"]
        }

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, target: str):
        """
        Parameters
        ----------
        target
            The target measure to modify, specified as ``"entity.measure"``
            (e.g. ``"cause.incidence_rate"``).
        """
        super().__init__()
        self.target = TargetString(target)

    def setup(self, builder: Builder) -> None:
        """Set up the component by reading configuration and registering the modifier.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.
        """
        self.config = builder.configuration[f"intervention_on_{self.target.name}"]
        builder.value.register_attribute_modifier(
            f"{self.target.name}.{self.target.measure}",
            modifier=self.intervention_effect,
            required_resources=["age"],
        )

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def intervention_effect(self, index: pd.Index, value: pd.Series) -> pd.Series:
        """Apply the configured absolute shift to the target measure.

        Replace the value for simulants within the configured age range with
        the configured ``target_value``. If ``target_value`` is ``"baseline"``,
        return the value unmodified.

        Parameters
        ----------
        index
            Index of the simulants to consider.
        value
            Current values of the target measure for the given simulants.

        Returns
        -------
            The modified target measure values.
        """
        if self.config["target_value"] != "baseline":
            affected_group_idx = self.population_view.get_population_index(
                index, query=f"{self.config['age_start']} <= age <= {self.config['age_end']}"
            )
            value.loc[affected_group_idx] = float(self.config["target_value"])
        return value
