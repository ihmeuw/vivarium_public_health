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
        return {
            f"intervention_on_{self.target.name}": self.CONFIGURATION_DEFAULTS["intervention"]
        }

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, target: str):
        super().__init__()
        self.target = TargetString(target)

    def setup(self, builder: Builder) -> None:
        self.config = builder.configuration[f"intervention_on_{self.target.name}"]
        builder.value.register_attribute_modifier(
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
            affected_group_idx = self.population_view.get_population_index(
                index, query=f"{self.config['age_start']} <= age <= {self.config['age_end']}"
            )
            value.loc[affected_group_idx] = float(self.config["target_value"])
        return value
