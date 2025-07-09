from vivarium_public_health.risks.exposure import Exposure


class Intervention(Exposure):
    """A component for modeling access to an intervention. The modeling and implementation of
    this component is similar to that of a risk factor where the risk is a loack or not having
    access to the intervention. Interventions should be configured with in the format of
    "intervention.intervention_name".

    """

    @property
    def measure_name(self) -> str:
        """The measure of the intervention access."""
        return "coverage"

    @property
    def exposed_category_name(self) -> str:
        """The name of the exposed category for this intervention."""
        return "covered"

    @property
    def unexposed_category_name(self) -> str:
        """The name of the unexposed category for this intervention."""
        return "uncovered"

    def __init__(self, intervention: str) -> None:
        super().__init__(intervention)
