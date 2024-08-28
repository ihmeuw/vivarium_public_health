"""
============
Simple Cause
============

This module contains tools for creating a minimal representation of a cause
as required by observers.

"""

from dataclasses import dataclass


@dataclass
class SimpleCause:
    """A simple dataclass to represent the bare minimum information needed by observers.

    It also includes a class method to convert a provided cause into a
    ``SimpleCause`` instance.

    """

    state_id: str
    """The state_id of the cause."""
    model: str
    """The model of the cause."""
    cause_type: str
    """The cause type of the cause."""

    @classmethod
    def create_from_specific_cause(cls, cause: type) -> "SimpleCause":
        """Create a SimpleCause instance from a more specific cause.

        Parameters
        ----------
        cause
            The cause to be converted into a SimpleCause instance.

        Returns
        -------
            A SimpleCause instance.
        """
        return cls(cause.state_id, cause.model, cause.cause_type)
