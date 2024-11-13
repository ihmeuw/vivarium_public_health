"""
=================
Magic Wand Models
=================

This module contains tools for making crude adjustments to rates in
multi-state lifetable simulations.

"""

from typing import Any

from vivarium import Component
from vivarium.framework.engine import Builder


class MortalityShift(Component):
    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: Builder) -> None:
        builder.value.register_value_modifier("mortality_rate", self.mortality_adjustment)

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def mortality_adjustment(self, index, rates):
        return rates * 0.5


class YLDShift(Component):
    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: Builder) -> None:
        builder.value.register_value_modifier("yld_rate", self.disability_adjustment)

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def disability_adjustment(self, index, rates):
        return rates * 0.5


class IncidenceShift(Component):
    #####################
    # Lifecycle methods #
    #####################
    def __init__(self, disease: str):
        super().__init__()
        self.disease = disease

    def setup(self, builder: Builder) -> None:
        builder.value.register_value_modifier(
            f"{self.disease}_intervention.incidence", self.incidence_adjustment
        )

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def incidence_adjustment(self, index, rates):
        return rates * 0.5


class ModifyAcuteDiseaseYLD(Component):
    ##############
    # Properties #
    ##############

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        return {
            "intervention": {
                self.disease: {
                    "yld_scale": 1.0,
                },
            }
        }

    #####################
    # Lifecycle methods #
    #####################
    def __init__(self, disease: str):
        super().__init__()
        self.disease = disease

    def setup(self, builder: Builder) -> None:
        self.config = builder.configuration
        self.scale = self.config.intervention[self.disease].yld_scale
        if self.scale < 0:
            raise ValueError(f"Invalid YLD scale: {self.scale}")
        builder.value.register_value_modifier(
            f"{self.disease}_intervention.yld_rate", self.disability_adjustment
        )

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def disability_adjustment(self, index, rates):
        return rates * self.scale


class ModifyAcuteDiseaseMortality(Component):
    ##############
    # Properties #
    ##############

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        return {
            "intervention": {
                self.disease: {
                    "mortality_scale": 1.0,
                },
            }
        }

    #####################
    # Lifecycle methods #
    #####################
    def __init__(self, disease: str):
        super().__init__()
        self.disease = disease

    def setup(self, builder: Builder) -> None:
        self.config = builder.configuration
        self.scale = self.config.intervention[self.disease].mortality_scale
        if self.scale < 0:
            raise ValueError(f"Invalid mortality scale: {self.scale}")
        builder.value.register_value_modifier(
            f"{self.disease}_intervention.excess_mortality", self.mortality_adjustment
        )

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def mortality_adjustment(self, index, rates):
        return rates * self.scale
