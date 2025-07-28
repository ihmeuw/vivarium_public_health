from typing import NamedTuple

from vivarium_public_health.exposure import Exposure
from vivarium_public_health.exposure.effect import ExposureEffect
from vivarium_public_health.utilities import EntityString, TargetString


class Intervention(Exposure):
    """A component for modeling access to an intervention. The modeling and implementation of
    this component is similar to that of a risk factor where the risk is a loack or not having
    access to the intervention. Interventions should be configured with in the format of
    "intervention.intervention_name".

    """

    @property
    def exposure_type(self) -> str:
        """The measure of the intervention access."""
        return "coverage"

    @property
    def dichotomous_exposure_category_names(self) -> NamedTuple:
        """The name of the exposed category for this intervention."""

        class __Categories(NamedTuple):
            exposed: str = "covered"
            unexposed: str = "uncovered"

        categories = __Categories()
        return categories


class InterventionEffect(ExposureEffect):
    """A component to model the effect of an intervention on an affected entity's target rate.

    This component can source data either from builder.data or from parameters
    supplied in the configuration.

    """

    @staticmethod
    def get_name(intervention: EntityString, target: TargetString) -> str:
        return f"intervention_effect.{intervention.name}_on_{target}"
