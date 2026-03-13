"""
==========================
Intervention Exposure Model
==========================

This module contains tools for modeling intervention exposures and their
effects on target measures.

"""

from vivarium_public_health.causal_factor.effect import CausalFactorEffect
from vivarium_public_health.causal_factor.exposure import CausalFactor


class Intervention(CausalFactor):
    """A model for an intervention defined by a dichotomous coverage value.

    This component can source data either from builder.data or from parameters
    supplied in the configuration. For an intervention named "my_intervention",
    the configuration could look like this:

    .. code-block:: yaml

       configuration:
           my_intervention:
               exposure: 0.5

    """

    VALID_ENTITY_TYPES = ["intervention"]

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, intervention: str):
        """

        Parameters
        ----------
        intervention
            The type and name of an intervention, specified as
            "type.name". Type is singular.
        """
        super().__init__(intervention)


class InterventionEffect(CausalFactorEffect):
    """A component to model the effect of an intervention on an affected
    entity's target measure.

    This component can source data either from builder.data or from parameters
    supplied in the configuration.

    For an intervention named 'my_intervention' that affects
    'affected_cause.incidence_rate', the configuration would look like:

    .. code-block:: yaml

       configuration:
            intervention_effect.my_intervention_on_cause.affected_cause.incidence_rate:
               data_sources:
                   relative_risk: 0.5

    """

    EXPOSURE_CLASS = Intervention

    ###############
    # Properties #
    ##############

    @property
    def name(self) -> str:
        return f"intervention_effect.{self.causal_factor.name}_on_{self.target}"

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, intervention: str, target: str):
        """

        Parameters
        ----------
        intervention
            Type and name of intervention, supplied in the form
            "intervention.intervention_name".
        target
            Type, name, and target measure of entity to be affected by the
            intervention, supplied in the form "entity_type.entity_name.measure"
            where entity_type should be singular (e.g., cause instead of causes).
        """
        super().__init__(intervention, target)
