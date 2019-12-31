"""
===================
The Model Menagerie
===================

This module contains a collection of frequently used parameterizations of
disease models.

"""
import pandas as pd

from vivarium_public_health.disease import SusceptibleState, RecoveredState, DiseaseState, DiseaseModel


def SI(cause: str) -> DiseaseModel:
    healthy = SusceptibleState(cause)
    infected = DiseaseState(cause)

    healthy.allow_self_transitions()
    healthy.add_transition(infected, source_data_type='rate')
    infected.allow_self_transitions()

    return DiseaseModel(cause, states=[healthy, infected])


def SIR(cause: str) -> DiseaseModel:
    healthy = SusceptibleState(cause)
    infected = DiseaseState(cause)
    recovered = RecoveredState(cause)

    healthy.allow_self_transitions()
    healthy.add_transition(infected, source_data_type='rate')
    infected.allow_self_transitions()
    infected.add_transition(recovered, source_data_type='rate')
    recovered.allow_self_transitions()

    return DiseaseModel(cause, states=[healthy, infected, recovered])


def SIS(cause: str) -> DiseaseModel:
    healthy = SusceptibleState(cause)
    infected = DiseaseState(cause)

    healthy.allow_self_transitions()
    healthy.add_transition(infected, source_data_type='rate')
    infected.allow_self_transitions()
    infected.add_transition(healthy, source_data_type='rate')

    return DiseaseModel(cause, states=[healthy, infected])


def SIS_fixed_duration(cause: str, duration: str) -> DiseaseModel:
    duration = pd.Timedelta(days=float(duration) // 1, hours=(float(duration) % 1) * 24.0)

    healthy = SusceptibleState(cause)
    infected = DiseaseState(cause, get_data_functions={'dwell_time': lambda _, __: duration})

    healthy.allow_self_transitions()
    healthy.add_transition(infected, source_data_type='rate')
    infected.add_transition(healthy)
    infected.allow_self_transitions()

    return DiseaseModel(cause, states=[healthy, infected])


def SIR_fixed_duration(cause: str, duration: str) -> DiseaseModel:
    duration = pd.Timedelta(days=float(duration) // 1, hours=(float(duration) % 1) * 24.0)

    healthy = SusceptibleState(cause)
    infected = DiseaseState(cause, get_data_functions={'dwell_time': lambda _, __: duration})
    recovered = RecoveredState(cause)

    healthy.allow_self_transitions()
    healthy.add_transition(infected, source_data_type='rate')
    infected.add_transition(recovered)
    infected.allow_self_transitions()
    recovered.allow_self_transitions()

    return DiseaseModel(cause, states=[healthy, infected, recovered])
    

def NeonatalSWC_without_incidence(cause):
    with_condition_data_functions = {'birth_prevalence':
                                     lambda cause, builder: builder.data.load(f"cause.{cause}.birth_prevalence")}

    healthy = SusceptibleState(cause)
    with_condition = DiseaseState(cause, get_data_functions=with_condition_data_functions)

    healthy.allow_self_transitions()
    with_condition.allow_self_transitions()

    return DiseaseModel(cause, states=[healthy, with_condition])


def NeonatalSWC_with_incidence(cause):
    with_condition_data_functions = {'birth_prevalence':
                                     lambda cause, builder: builder.data.load(f"cause.{cause}.birth_prevalence")}

    healthy = SusceptibleState(cause)
    with_condition = DiseaseState(cause, get_data_functions=with_condition_data_functions)

    healthy.allow_self_transitions()
    healthy.add_transition(with_condition, source_data_type='rate')
    with_condition.allow_self_transitions()

    return DiseaseModel(cause, states=[healthy, with_condition])
