"""
==========================
Simple Intervention Models
==========================

This module contains simple intervention models that work at the population
level by providing direct shifts to epidemiological measures.

"""

from typing import Any, Dict, List, Optional

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
    def configuration_defaults(self) -> Dict[str, Any]:
        return {
            f"intervention_on_{self.target.name}": self.CONFIGURATION_DEFAULTS["intervention"]
        }

    @property
    def columns_required(self) -> Optional[List[str]]:
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
            requires_columns=["age"],
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
