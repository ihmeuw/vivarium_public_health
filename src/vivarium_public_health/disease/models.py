import pandas as pd

from vivarium_public_health.disease import (SusceptibleState, ExcessMortalityState, RecoveredState,
                                            DiseaseState, DiseaseModel)
from vivarium_inputs.utilities import DataMissingError
from vivarium_public_health.dataset_manager.artifact import ArtifactException


def get_aggregate_disability_weight(cause: str, builder) -> pd.DataFrame:
    """Calculates the cause-level disability weight as the sum of the causes's sequelae's disability weights
    weighted by their prevalences.

    Parameters
    ----------
    cause:
        A cause name
    builder:
        A vivarium builder object

    Returns
    -------
        The cause-level disability weight, varying by year, age and sex.
    """
    sequelae = builder.data.load(f"cause.{cause}.sequelae")
    aggregate_dw = None
    for s in sequelae:
        prevalence = builder.data.load(f"sequela.{s}.prevalence")
        prevalence.drop(['sequela_id'], axis=1, inplace=True)
        try:
            disability_weight = builder.data.load(f"sequela.{s}.disability_weight")
            assert disability_weight.shape[0] == 1
            disability_weight = float(disability_weight.value)
        except (DataMissingError, ArtifactException):
            disability_weight = 0.0
        prevalence['value'] *= disability_weight
        prevalence.set_index(['sex', 'age_group_start', 'age_group_end',
                              'year_start', 'year_end'], inplace=True)
        if aggregate_dw is None:
            aggregate_dw = prevalence.copy()
        else:
            aggregate_dw += prevalence

    return aggregate_dw.reset_index()


class SI:

    def __init__(self, cause: str):
        self.cause = cause

    def setup(self, builder):
        only_morbid = builder.data.load(f'cause.{self.cause}.restrictions')['yld_only']

        healthy = SusceptibleState(self.cause)
        healthy.allow_self_transitions()

        get_data_functions = {}
        if only_morbid:
            infected = DiseaseState(self.cause,
                                    get_data_functions={'disability_weight': get_aggregate_disability_weight})
            get_data_functions['csmr'] = lambda _, __: None  # DiseaseModel will try to pull not provided
        else:
            infected = ExcessMortalityState(self.cause,
                                            get_data_functions={'disability_weight': get_aggregate_disability_weight})
        infected.allow_self_transitions()

        healthy.add_transition(infected, source_data_type='rate')

        builder.components.add_components([DiseaseModel(self.cause, states=[healthy, infected],
                                                        get_data_functions=get_data_functions)])


class SIR:

    def __init__(self, cause: str):
        self.cause = cause

    def setup(self, builder):
        only_morbid = builder.data.load(f'cause.{self.cause}.restrictions')['yld_only']

        healthy = SusceptibleState(self.cause)
        healthy.allow_self_transitions()

        get_data_functions = {}
        if only_morbid:
            infected = DiseaseState(self.cause,
                                    get_data_functions={'disability_weight': get_aggregate_disability_weight})
            get_data_functions['csmr'] = lambda _, __: None  # DiseaseModel will try to pull not provided
        else:
            infected = ExcessMortalityState(self.cause,
                                            get_data_functions={'disability_weight': get_aggregate_disability_weight})
        infected.allow_self_transitions()

        recovered = RecoveredState(self.cause)
        recovered.allow_self_transitions()

        healthy.add_transition(infected, source_data_type='rate')
        infected.add_transition(recovered, source_data_type='rate')

        builder.components.add_components([DiseaseModel(self.cause, states=[healthy, infected, recovered],
                                                        get_data_functions=get_data_functions)])


class SIS:

    def __init__(self, cause):
        self.cause = cause

    def setup(self, builder):
        only_morbid = builder.data.load(f'cause.{self.cause}.restrictions')['yld_only']

        healthy = SusceptibleState(self.cause)
        healthy.allow_self_transitions()

        get_data_functions = {}
        if only_morbid:
            infected = DiseaseState(self.cause,
                                    get_data_functions={'disability_weight': get_aggregate_disability_weight})
            get_data_functions['csmr'] = lambda _, __: None  # DiseaseModel will try to pull not provided
        else:
            infected = ExcessMortalityState(self.cause,
                                            get_data_functions={'disability_weight': get_aggregate_disability_weight})
        infected.allow_self_transitions()

        healthy.add_transition(infected, source_data_type='rate')
        infected.add_transition(healthy, source_data_type='rate')

        builder.components.add_components([DiseaseModel(self.cause, states=[healthy, infected],
                                                        get_data_functions=get_data_functions)])


class SIS_fixed_duration:

    def __init__(self, cause, duration):
        self.cause = cause
        if not isinstance(duration, pd.Timedelta):
            self.duration = pd.Timedelta(days=float(duration) // 1, hours=float(duration) % 1)
        else:
            self.duration = duration

    def setup(self, builder):
        only_morbid = builder.data.load(f'cause.{self.cause}.restrictions')['yld_only']

        healthy = SusceptibleState(self.cause)
        healthy.allow_self_transitions()

        get_data_functions = {}
        if only_morbid:
            infected = DiseaseState(self.cause,
                                    get_data_functions={'disability_weight': get_aggregate_disability_weight,
                                                        'dwell_time': lambda _, __: self.duration})
            get_data_functions['csmr'] = lambda _, __: None  # DiseaseModel will try to pull not provided
        else:
            infected = ExcessMortalityState(self.cause,
                                            get_data_functions={'disability_weight': get_aggregate_disability_weight,
                                                                'dwell_time': lambda _, __: self.duration})
        infected.allow_self_transitions()

        healthy.add_transition(infected, source_data_type='rate')
        infected.add_transition(healthy)

        builder.components.add_components([DiseaseModel(self.cause, states=[healthy, infected],
                                                        get_data_functions=get_data_functions)])


class neonatal:

    def __init__(self, cause):
        self.cause = cause

    def setup(self, builder):

        only_morbid = builder.data.load(f'cause.{self.cause}.restricitons')['yld_only']

        healthy = SusceptibleState(self.cause)
        healthy.allow_self_transitions()

        get_data_functions = {}
        if only_morbid:
            with_condition = DiseaseState(self.cause,
                                          get_data_functions={'disability_weight': get_aggregate_disability_weight})
            get_data_functions['csmr'] = lambda _, __: None  # DiseaseModel will try to pull not provided
        else:
            with_condition = ExcessMortalityState(self.cause,
                                                  get_data_functions={'disability_weight': get_aggregate_disability_weight})
        with_condition.allow_self_transitions()

        # TODO: some neonatal causes (e.g. sepsis) have incidence and remission at least at the MEID level
        # healthy.add_transition(with_condition, source_data_type='rate')
        # with_condition.add_transition(healthy, source_data_type='rate')

        builder.components.add_components([DiseaseModel(self.cause, states=[healthy, with_condition],
                                                        get_data_functions=get_data_functions)])






