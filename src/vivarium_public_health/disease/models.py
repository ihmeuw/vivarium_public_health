"""
===================
The Model Menagerie
===================

This module contains a collection of frequently used parameterizations of
disease models.

"""

import pandas as pd

from vivarium_public_health.disease.model import DiseaseModel
from vivarium_public_health.disease.state import (
    DiseaseState,
    RecoveredState,
    SusceptibleState,
)


def SI(cause: str) -> DiseaseModel:
    """Create a Susceptible-Infected disease model.

    Parameters
    ----------
    cause
        The name of the cause of disease.

    Returns
    -------
        A configured SI disease model.
    """
    healthy = SusceptibleState(cause)
    infected = DiseaseState(cause)

    healthy.add_rate_transition(infected)

    return DiseaseModel(cause, states=[healthy, infected])


def SIR(cause: str) -> DiseaseModel:
    """Create a Susceptible-Infected-Recovered disease model.

    Parameters
    ----------
    cause
        The name of the cause of disease.

    Returns
    -------
        A configured SIR disease model.
    """
    healthy = SusceptibleState(cause)
    infected = DiseaseState(cause)
    recovered = RecoveredState(cause)

    healthy.add_rate_transition(infected)
    infected.add_rate_transition(recovered)

    return DiseaseModel(cause, states=[healthy, infected, recovered])


def SIS(cause: str) -> DiseaseModel:
    """Create a Susceptible-Infected-Susceptible disease model.

    Parameters
    ----------
    cause
        The name of the cause of disease.

    Returns
    -------
        A configured SIS disease model.
    """
    healthy = SusceptibleState(cause)
    infected = DiseaseState(cause)

    healthy.add_rate_transition(infected)
    infected.add_rate_transition(healthy)

    return DiseaseModel(cause, states=[healthy, infected])


def SIS_fixed_duration(cause: str, duration: str) -> DiseaseModel:
    """Create an SIS disease model with a fixed infection duration.

    Parameters
    ----------
    cause
        The name of the cause of disease.
    duration
        The duration of infection in days.

    Returns
    -------
        A configured SIS disease model with fixed duration.
    """
    duration = pd.Timedelta(days=float(duration) // 1, hours=(float(duration) % 1) * 24.0)

    healthy = SusceptibleState(cause)
    infected = DiseaseState(cause, dwell_time=duration)

    healthy.add_rate_transition(infected)
    infected.add_dwell_time_transition(healthy)

    return DiseaseModel(cause, states=[healthy, infected])


def SIR_fixed_duration(cause: str, duration: str) -> DiseaseModel:
    """Create an SIR disease model with a fixed infection duration.

    Parameters
    ----------
    cause
        The name of the cause of disease.
    duration
        The duration of infection in days.

    Returns
    -------
        A configured SIR disease model with fixed duration.
    """
    duration = pd.Timedelta(days=float(duration) // 1, hours=(float(duration) % 1) * 24.0)

    healthy = SusceptibleState(cause)
    infected = DiseaseState(cause, dwell_time=duration)
    recovered = RecoveredState(cause)

    healthy.add_rate_transition(infected)
    infected.add_dwell_time_transition(recovered)

    return DiseaseModel(cause, states=[healthy, infected, recovered])


def NeonatalSWC_without_incidence(cause: str) -> DiseaseModel:
    """Create a neonatal model with birth prevalence but no incidence.

    Parameters
    ----------
    cause
        The name of the cause of disease.

    Returns
    -------
        A configured neonatal disease model without incidence.
    """
    healthy = SusceptibleState(cause)
    with_condition = DiseaseState(cause, birth_prevalence=f"cause.{cause}.birth_prevalence")

    return DiseaseModel(cause, states=[healthy, with_condition])


def NeonatalSWC_with_incidence(cause: str) -> DiseaseModel:
    """Create a neonatal model with both birth prevalence and incidence.

    Parameters
    ----------
    cause
        The name of the cause of disease.

    Returns
    -------
        A configured neonatal disease model with incidence.
    """
    healthy = SusceptibleState(cause)
    with_condition = DiseaseState(cause, birth_prevalence=f"cause.{cause}.birth_prevalence")

    healthy.add_rate_transition(with_condition)

    return DiseaseModel(cause, states=[healthy, with_condition])
