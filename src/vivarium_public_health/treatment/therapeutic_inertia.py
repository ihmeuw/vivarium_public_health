"""
=========================
Therapeutic Inertia Model
=========================

Model :term:`therapeutic inertia <Therapeutic Inertia>`, the variety of reasons
why a treatment algorithm might deviate from clinical guidelines.

"""

import pandas as pd
import scipy.stats
from vivarium import Component
from vivarium.framework.engine import Builder


class TherapeuticInertia(Component):
    """Produce a population-level :term:`therapeutic inertia <Therapeutic Inertia>` value.

    At setup a single scalar therapeutic inertia value is drawn from a
    triangular distribution parameterized by ``triangle_min``,
    ``triangle_max``, and ``triangle_mode``. This value represents the
    probability that treatment is *not* escalated during a healthcare visit
    and is exposed via the ``therapeutic_inertia`` pipeline.

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
        """Set up the component by drawing a therapeutic inertia value and registering the pipeline.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.
        """
        self.therapeutic_inertia_parameters = builder.configuration.therapeutic_inertia

        self._therapeutic_inertia = self.initialize_therapeutic_inertia(builder)
        ti_source = lambda index: pd.Series(self._therapeutic_inertia, index=index)
        builder.value.register_attribute_producer(
            "therapeutic_inertia", source=ti_source, component=self
        )

    #################
    # Setup methods #
    #################

    def initialize_therapeutic_inertia(self, builder: Builder) -> float:
        """Draw a single therapeutic inertia value from the configured triangular distribution.

        The triangular distribution is parameterized by ``triangle_min``,
        ``triangle_max``, and ``triangle_mode`` from the component's
        configuration. The resulting scalar is used for the entire simulation.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.

        Returns
        -------
            A scalar therapeutic inertia value drawn from the triangular
            distribution.
        """
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
