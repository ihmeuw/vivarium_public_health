"""
===================
Intervention Models
===================

This module contains tools for modeling interventions in multi-state lifetable
simulations.

"""

from typing import Any

from vivarium import Component
from vivarium.framework.engine import Builder


class ModifyAllCauseMortality(Component):
    """Interventions that modify the all-cause mortality rate."""

    ##############
    # Properties #
    ##############

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        return {
            "intervention": {
                self.intervention: {
                    "scale": 1.0,
                },
            }
        }

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, intervention: str):
        super().__init__()
        self.intervention = intervention

    def setup(self, builder: Builder) -> None:
        self.config = builder.configuration
        self.scale = self.config.intervention[self.intervention]["scale"]
        if self.scale < 0:
            raise ValueError("Invalid scale: {}".format(self.scale))
        builder.value.register_value_modifier("mortality_rate", self.mortality_adjustment)

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def mortality_adjustment(self, index, rates):
        return rates * self.scale


class ModifyDiseaseRate(Component):
    """Interventions that modify a rate associated with a chronic disease."""

    ##############
    # Properties #
    ##############

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        return {
            "intervention": {
                self.intervention: {
                    self._scale_name: 1.0,
                },
            }
        }

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, intervention: str, disease: str, rate: str):
        super().__init__()
        self.intervention = intervention
        self.disease = disease
        self.rate = rate
        self._scale_name = f"{self.disease}_{self.rate}_scale"

    def setup(self, builder: Builder) -> None:
        self.config = builder.configuration
        # NOTE: this will be replaced by an (age, sex, year) lookup-table.
        self.scale = self.config.intervention[self.intervention][self._scale_name]
        if self.scale < 0:
            raise ValueError("Invalid scale: {}".format(self.scale))
        rate_name = "{}_intervention.{}".format(self.disease, self.rate)
        builder.value.register_value_modifier(rate_name, self.adjust_rate)

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def adjust_rate(self, index, rates):
        return rates * self.scale


class ModifyDiseaseIncidence(ModifyDiseaseRate):
    """Interventions that modify a disease incidence rate, based on a PIF lookup
    table.

    """

    def __init__(self, intervention: str, disease: str):
        super().__init__(intervention=intervention, disease=disease, rate="incidence")


class ModifyDiseaseMortality(ModifyDiseaseRate):
    """Interventions that modify a disease fatality rate, based on a PIF lookup
    table.

    """

    def __init__(self, intervention: str, disease: str):
        super().__init__(intervention=intervention, disease=disease, rate="excess_mortality")


class ModifyDiseaseMorbidity(ModifyDiseaseRate):
    """Interventions that modify a disease disability rate, based on a PIF lookup
    table.

    """

    def __init__(self, intervention: str, disease: str):
        super().__init__(intervention=intervention, disease=disease, rate="yld_rate")


class ModifyAcuteDiseaseIncidence(Component):
    """Interventions that modify an acute disease incidence rate.

    Notes
    -----
    This intervention will simply modify both the disability rate
    and the mortality rate for the chosen acute disease.

    """

    ##############
    # Properties #
    ##############

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        return {
            "intervention": {
                self.intervention: {
                    "incidence_scale": 1.0,
                },
            }
        }

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, intervention: str):
        super().__init__()
        self.intervention = intervention

    def setup(self, builder: Builder) -> None:
        self.config = builder.configuration
        self.scale = self.config.intervention[self.intervention].incidence_scale
        if self.scale < 0:
            raise ValueError("Invalid incidence scale: {}".format(self.scale))
        yld_rate = "{}_intervention.yld_rate".format(self.intervention)
        builder.value.register_value_modifier(yld_rate, self.rate_adjustment)
        mort_rate = "{}_intervention.excess_mortality".format(self.intervention)
        builder.value.register_value_modifier(mort_rate, self.rate_adjustment)

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def rate_adjustment(self, index, rates):
        return rates * self.scale


class ModifyAcuteDiseaseMorbidity(Component):
    """Interventions that modify an acute disease disability rate."""

    ##############
    # Properties #
    ##############

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        return {
            "intervention": {
                self.intervention: {
                    "yld_scale": 1.0,
                },
            }
        }

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, intervention: str):
        super().__init__()
        self.intervention = intervention

    def setup(self, builder: Builder) -> None:
        self.config = builder.configuration
        self.scale = self.config.intervention[self.intervention].yld_scale
        if self.scale < 0:
            raise ValueError("Invalid YLD scale: {}".format(self.scale))
        rate = "{}_intervention.yld_rate".format(self.intervention)
        builder.value.register_value_modifier(rate, self.disability_adjustment)

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def disability_adjustment(self, index, rates):
        return rates * self.scale


class ModifyAcuteDiseaseMortality(Component):
    """Interventions that modify an acute disease fatality rate."""

    ##############
    # Properties #
    ##############

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        return {
            "intervention": {
                self.intervention: {
                    "mortality_scale": 1.0,
                },
            }
        }

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, intervention: str):
        super().__init__()
        self.intervention = intervention

    def setup(self, builder: Builder) -> None:
        self.config = builder.configuration
        self.scale = self.config.intervention[self.intervention].mortality_scale
        if self.scale < 0:
            raise ValueError("Invalid mortality scale: {}".format(self.scale))
        rate = "{}_intervention.excess_mortality".format(self.intervention)
        builder.value.register_value_modifier(rate, self.mortality_adjustment)

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def mortality_adjustment(self, index, rates):
        return rates * self.scale


class TobaccoFreeGeneration(Component):
    """Eradicate tobacco uptake at some point in time."""

    ##############
    # Properties #
    ##############

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        return {
            "tobacco_free_generation": {
                "year": 2020,
            },
        }

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self):
        super().__init__()
        self.exposure = "tobacco"

    def setup(self, builder: Builder) -> None:
        self.year = builder.configuration["tobacco_free_generation"].year
        self.clock = builder.time.clock()
        rate_name = "{}_intervention.incidence".format(self.exposure)
        builder.value.register_value_modifier(rate_name, self.adjust_rate)

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def adjust_rate(self, index, rates):
        this_year = self.clock().year
        if this_year >= self.year:
            return 0.0 * rates
        else:
            return rates


class TobaccoEradication(Component):
    """Eradicate all tobacco use at some point in time."""

    ##############
    # Properties #
    ##############

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        return {
            "tobacco_eradication": {
                "year": 2020,
            },
        }

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self):
        super().__init__()
        self.exposure = "tobacco"

    def setup(self, builder: Builder) -> None:
        self.year = builder.configuration["tobacco_eradication"].year
        self.clock = builder.time.clock()
        inc_rate_name = "{}_intervention.incidence".format(self.exposure)
        builder.value.register_value_modifier(inc_rate_name, self.adjust_inc_rate)
        rem_rate_name = "{}_intervention.remission".format(self.exposure)
        builder.value.register_value_modifier(rem_rate_name, self.adjust_rem_rate)

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def adjust_inc_rate(self, index, rates):
        this_year = self.clock().year
        if this_year >= self.year:
            return 0.0 * rates
        else:
            return rates

    def adjust_rem_rate(self, index, rates):
        this_year = self.clock().year
        if this_year >= self.year:
            rates[:] = 1.0
        return rates
