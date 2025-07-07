from vivarium_public_health.risks.health_factor import HealthFactor


class InterventionAccess(HealthFactor):
    """A component for modeling access to an intervention. The modeling and implementation of
    this component is similar to that of a risk factor where the risk is a loack or not having
    access to the intervention.

    """

    def __init__(self, health_factor, level_type="coverage"):
        super().__init__(health_factor, level_type)
