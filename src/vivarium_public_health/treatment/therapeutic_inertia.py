"""
=========================
Therapeutic Inertia Model
=========================

This module contains a model for therapeutic inertia which represents the
variety of reasons why a treatment algorithm might deviate from guidelines.

"""

import pandas as pd
import scipy.stats
from vivarium import Component
from vivarium.framework.engine import Builder


class TherapeuticInertia(Component):
    """Expose a therapeutic inertia pipeline that defines
    a population-level therapeutic inertia.

    This is the probability of treatment during a healthcare visit.
    """

    CONFIGURATION_DEFAULTS = {
        "therapeutic_inertia": {
            "triangle_min": 0.65,
            "triangle_max": 0.9,
            "triangle_mode": 0.875,
        }
    }

    def __str__(self):
        return (
            f"TherapeuticInertia(triangle_min={self.therapeutic_inertia_parameters.triangle_min}, "
            f"triangle_max={self.therapeutic_inertia_parameters.triangle_max}, "
            f"triangle_mode={self.therapeutic_inertia_parameters.triangle_mode})"
        )

    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: Builder) -> None:
        self.therapeutic_inertia_parameters = builder.configuration.therapeutic_inertia

        self._therapeutic_inertia = self.initialize_therapeutic_inertia(builder)
        ti_source = lambda index: pd.Series(self._therapeutic_inertia, index=index)
        self.therapeutic_inertia = builder.value.register_value_producer(
            "therapeutic_inertia", source=ti_source
        )

    #################
    # Setup methods #
    #################

    def initialize_therapeutic_inertia(self, builder):
        triangle_min = self.therapeutic_inertia_parameters.triangle_min
        triangle_max = self.therapeutic_inertia_parameters.triangle_max
        triangle_mode = self.therapeutic_inertia_parameters.triangle_mode

        # convert to scipy params
        loc = triangle_min
        scale = triangle_max - triangle_min
        if scale == 0:
            c = 0
        else:
            c = (triangle_mode - loc) / scale

        seed = builder.randomness.get_seed(self.name)
        therapeutic_inertia = scipy.stats.triang(c, loc=loc, scale=scale).rvs(
            random_state=seed
        )

        return therapeutic_inertia
