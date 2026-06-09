"""Exception classes for the disease modeling package."""

from vivarium.engine.exceptions import VivariumError


class DiseaseModelError(VivariumError):
    """Error raised when a disease model is improperly configured."""

    pass
