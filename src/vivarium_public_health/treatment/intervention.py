from typing import NamedTuple

from vivarium_public_health.exposure import Exposure
from vivarium_public_health.exposure.effect import ExposureEffect
from vivarium_public_health.utilities import EntityString, TargetString


class Intervention(Exposure):
    """A model for an intervention defined by coverage (access to intervention).

    This component models access to an intervention as a dichotomous exposure where
    simulants are either covered or uncovered by the intervention.

    For example,

    #. vaccination coverage where simulants are either vaccinated (covered) or
       unvaccinated (uncovered).
    #. treatment access where simulants either have access to treatment (covered)
       or do not have access (uncovered).

    This component can source data either from a key in an Artifact
    ("intervention.intervention_name.coverage") or from parameters
    supplied in the configuration. If data is derived from the configuration, it
    must be an integer or float expressing the desired coverage level or a
    covariate name that is intended to be used as a proxy. For example, for an
    intervention named "intervention", the configuration could look like this:

    .. code-block:: yaml

       configuration:
           intervention.intervention_name:
               coverage: 0.8

    Interventions should be configured with names in the format of
    "intervention.intervention_name".

    """

    @property
    def exposure_type(self) -> str:
        """The measure of the intervention access."""
        return "coverage"

    @property
    def dichotomous_exposure_category_names(self) -> NamedTuple:
        """The name of the covered and uncovered categories for this intervention."""

        class __Categories(NamedTuple):
            exposed: str = "covered"
            unexposed: str = "uncovered"

        categories = __Categories()
        return categories


class InterventionEffect(ExposureEffect):
    """A component to model the effect of an intervention on an affected entity's target rate.

    This component models how intervention coverage affects the rate of some target
    entity (e.g., disease incidence, mortality, disability). The effect is typically
    protective, reducing the target rate for covered simulants compared to uncovered
    simulants.

    This component can source data either from builder.data or from parameters
    supplied in the configuration. The data should specify the relative risk or
    rate ratio associated with intervention coverage.

    For example, an intervention effect might model how vaccination coverage affects
    disease incidence, where vaccinated individuals have a lower risk of disease
    compared to unvaccinated individuals.

    For an exposure named 'exposure_name' that affects  'affected_entity' and 'affected_cause',
    the configuration would look like:

    .. code-block:: yaml

       configuration:
            intervention_effect.exposure_name_on_affected_target:
               exposure_parameters: 2
               incidence_rate: 10

    """

    def get_name(self, intervention: EntityString, target: TargetString) -> str:
        return f"intervention_effect.{intervention.name}_on_{target}"
