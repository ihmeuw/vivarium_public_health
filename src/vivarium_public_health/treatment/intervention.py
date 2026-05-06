"""
===========================
Intervention Exposure Model
===========================

This module contains tools for modeling intervention exposures and their
effects on target measures.

"""

from vivarium_public_health.causal_factor.effect import MultiplicativeEffect
from vivarium_public_health.causal_factor.exposure import CausalFactor
from vivarium_public_health.utilities import EntityString, TargetString


class Intervention(CausalFactor):
    """A model for an intervention defined by a dichotomous coverage value.

    This is a specialization of :class:`~vivarium_public_health.causal_factor.exposure.CausalFactor`
    restricted to ``"intervention"`` entity types. It can source data either
    from the artifact or from parameters supplied in the configuration. For an
    intervention named ``my_intervention``, the configuration could look like:

    .. code-block:: yaml

        configuration:
            my_intervention:
                exposure: 0.5

    """

    VALID_ENTITY_TYPES = ["intervention"]

    @property
    def intervention(self) -> str:
        """The type and name of the intervention, specified as "type.name". Type is singular."""
        return self.causal_factor

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, intervention: str):
        """
        Parameters
        ----------
        intervention
            The type and name of an intervention, specified as
            ``"type.name"``. Type is singular.
        """
        super().__init__(intervention)


class InterventionEffect(MultiplicativeEffect):
    """A model for the effect of an intervention on an affected
    entity's target measure.

    This is a specialization of
    :class:`~vivarium_public_health.causal_factor.effect.MultiplicativeEffect`
    for interventions. It can source relative risk and population attributable
    fraction data from the artifact or from scalar configuration parameters.

    For an intervention named ``my_intervention`` that affects
    ``affected_cause.incidence_rate``, the configuration would look like:

    .. code-block:: yaml

        configuration:
            intervention_effect.my_intervention_on_cause.affected_cause.incidence_rate:
                data_sources:
                    relative_risk: 0.5

    """

    EXPOSURE_CLASS = Intervention

    ##############
    # Properties #
    ##############

    @staticmethod
    def get_name(intervention: EntityString, target: TargetString) -> str:
        """The name of this intervention effect component."""
        return f"intervention_effect.{intervention.name}_on_{target}"

    @property
    def intervention(self) -> str:
        """The type and name of the intervention, specified as "type.name". Type is singular."""
        return self.causal_factor

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, intervention: str, target: str):
        """
        Parameters
        ----------
        intervention
            Type and name of intervention, supplied in the form
            ``"intervention.intervention_name"``.
        target
            Type, name, and target measure of entity to be affected by the
            intervention, supplied in the form
            ``"entity_type.entity_name.measure"`` where ``entity_type``
            should be singular (e.g. ``cause`` instead of ``causes``).
        """
        super().__init__(intervention, target)
