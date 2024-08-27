from dataclasses import dataclass


@dataclass
class SimpleCause:
    """A simple dataclass to represent the bare minimum information needed
    for observers, e.g. 'all_causes' as a cause of disability.

    It also includes a class method to convert a provided disease state into a
    ``SimpleCause`` instance.

    """

    state_id: str
    """The state_id of the cause."""
    model: str
    """The model of the cause."""
    cause_type: str
    """The cause type of the cause."""

    @classmethod
    def create_from_disease_state(cls, disease_state: type) -> "SimpleCause":
        """Create a SimpleCause instance from a"""
        return cls(disease_state.state_id, disease_state.model, disease_state.cause_type)
