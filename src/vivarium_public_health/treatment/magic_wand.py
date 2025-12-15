"""
==========================
Simple Intervention Models
==========================

This module contains simple intervention models that work at the population
level by providing direct shifts to epidemiological measures.

"""

from typing import Any

from vivarium import Component
from vivarium.framework.engine import Builder

from vivarium_public_health.utilities import TargetString


class AbsoluteShift(Component):
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

    @property
    def columns_required(self) -> list[str] | None:
        return ["age"]

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, target: str):
        super().__init__()
        self.target = TargetString(target)

    def setup(self, builder: Builder) -> None:
        self.config = builder.configuration[f"intervention_on_{self.target.name}"]
        builder.value.register_value_modifier(
            f"{self.target.name}.{self.target.measure}",
            modifier=self.intervention_effect,
            component=self,
            required_resources=["age"],
        )

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def intervention_effect(self, index, value):
        if self.config["target_value"] != "baseline":
            pop = self.population_view.get(index)
            affected_group = pop[
                pop.age.between(self.config["age_start"], self.config["age_end"])
            ]
            value.loc[affected_group.index] = float(self.config["target_value"])
        return value
