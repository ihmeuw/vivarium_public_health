from vivarium_public_health.risks.health_factor import HealthFactor


class Intervention(HealthFactor):
    def __init__(self, health_factor, level_type="coverage"):
        super().__init__(health_factor, level_type)
