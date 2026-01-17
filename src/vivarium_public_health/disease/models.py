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
    healthy = SusceptibleState(cause, allow_self_transition=True)
    infected = DiseaseState(cause, allow_self_transition=True)

    healthy.add_rate_transition(infected)

    return DiseaseModel(cause, states=[healthy, infected])


def SIR(cause: str) -> DiseaseModel:
    healthy = SusceptibleState(cause, allow_self_transition=True)
    infected = DiseaseState(cause, allow_self_transition=True)
    recovered = RecoveredState(cause, allow_self_transition=True)

    healthy.add_rate_transition(infected)
    infected.add_rate_transition(recovered)

    return DiseaseModel(cause, states=[healthy, infected, recovered])


def SIS(cause: str) -> DiseaseModel:
    healthy = SusceptibleState(cause, allow_self_transition=True)
    infected = DiseaseState(cause, allow_self_transition=True)

    healthy.add_rate_transition(infected)
    infected.add_rate_transition(healthy)

    return DiseaseModel(cause, states=[healthy, infected])


def SIS_fixed_duration(cause: str, duration: str) -> DiseaseModel:
    duration = pd.Timedelta(days=float(duration) // 1, hours=(float(duration) % 1) * 24.0)

    healthy = SusceptibleState(cause, allow_self_transition=True)
    infected = DiseaseState(cause, dwell_time=duration, allow_self_transition=True)

    healthy.add_rate_transition(infected)
    infected.add_dwell_time_transition(healthy)

    return DiseaseModel(cause, states=[healthy, infected])


def SIR_fixed_duration(cause: str, duration: str) -> DiseaseModel:
    duration = pd.Timedelta(days=float(duration) // 1, hours=(float(duration) % 1) * 24.0)

    healthy = SusceptibleState(cause, allow_self_transition=True)
    infected = DiseaseState(cause, dwell_time=duration, allow_self_transition=True)
    recovered = RecoveredState(cause, allow_self_transition=True)

    healthy.add_rate_transition(infected)
    infected.add_dwell_time_transition(recovered)

    return DiseaseModel(cause, states=[healthy, infected, recovered])


def NeonatalSWC_without_incidence(cause):
    healthy = SusceptibleState(cause, allow_self_transition=True)
    with_condition = DiseaseState(
        cause, birth_prevalence=f"cause.{cause}.birth_prevalence", allow_self_transition=True
    )

    return DiseaseModel(cause, states=[healthy, with_condition])


def NeonatalSWC_with_incidence(cause):
    healthy = SusceptibleState(cause, allow_self_transition=True)
    with_condition = DiseaseState(
        cause, birth_prevalence=f"cause.{cause}.birth_prevalence", allow_self_transition=True
    )

    healthy.add_rate_transition(with_condition)

    return DiseaseModel(cause, states=[healthy, with_condition])
