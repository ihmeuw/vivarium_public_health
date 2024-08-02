from dataclasses import dataclass


@dataclass
class SimpleCause:
    """A simple dataclass to represent the bare minimum information needed
    for observers, e.g. 'all_causes' as a cause of disability. It also
    includes a class method to coerce a provided disease state into a
    ``SimpleCause`` instance.
    """

    state_id: str
    model: str
    cause_type: str

    @classmethod
    def create_from_disease_state(cls, disease_state):
        return cls(disease_state.state_id, disease_state.model, disease_state.cause_type)
