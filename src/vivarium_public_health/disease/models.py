import pandas as pd

from vivarium_public_health.disease import (SusceptibleState, ExcessMortalityState, RecoveredState,
                                            DiseaseState, DiseaseModel)


class SI:

    def __init__(self, cause: str):
        self.cause = cause

    def setup(self, builder):
        only_morbid = builder.data.load(f'cause.{self.cause}.restrictions')['yld_only']

        healthy = SusceptibleState(self.cause)
        healthy.allow_self_transitions()

        get_data_functions = {}
        if only_morbid:
            infected = DiseaseState(self.cause)
            get_data_functions['csmr'] = lambda _, __: None
        else:
            infected = ExcessMortalityState(self.cause) 
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
            infected = DiseaseState(self.cause)
            get_data_functions['csmr'] = lambda _, __: None
        else:
            infected = ExcessMortalityState(self.cause)

        infected.allow_self_transitions()

        recovered = RecoveredState(self.cause)
        recovered.allow_self_transitions()

        healthy.add_transition(infected, source_data_type='rate')
        infected.add_transition(recovered, source_data_type='rate')

        builder.components.add_components([DiseaseModel(self.cause, states=[healthy, infected, recovered],
                                                        get_data_functions=get_data_functions)])


class SIS:

    def __init__(self, cause: str):
        self.cause = cause

    def setup(self, builder):
        only_morbid = builder.data.load(f'cause.{self.cause}.restrictions')['yld_only']

        healthy = SusceptibleState(self.cause)
        healthy.allow_self_transitions()

        get_data_functions = {}
        if only_morbid:
            infected = DiseaseState(self.cause)
            get_data_functions['csmr'] = lambda _, __: None
        else:
            infected = ExcessMortalityState(self.cause)

        infected.allow_self_transitions()

        healthy.add_transition(infected, source_data_type='rate')
        infected.add_transition(healthy, source_data_type='rate')

        builder.components.add_components([DiseaseModel(self.cause, states=[healthy, infected],
                                                        get_data_functions=get_data_functions)])


class SIS_fixed_duration:

    def __init__(self, cause: str, duration):
        """

        Parameters
        ----------
        cause
        duration
        """
        self.cause = cause
        if not isinstance(duration, pd.Timedelta):
            self.duration = pd.Timedelta(days=float(duration) // 1, hours=(float(duration) % 1) * 24.0)
        else:
            self.duration = duration

    def setup(self, builder):
        only_morbid = builder.data.load(f'cause.{self.cause}.restrictions')['yld_only']

        healthy = SusceptibleState(self.cause)
        healthy.allow_self_transitions()

        get_data_functions = {}
        if only_morbid:
            infected = DiseaseState(self.cause,
                                    get_data_functions={'dwell_time': lambda _, __: self.duration})
            get_data_functions['csmr'] = lambda _, __: None
        else:
            infected = ExcessMortalityState(self.cause,
                                            get_data_functions={'dwell_time': lambda _, __: self.duration})
        infected.allow_self_transitions()

        healthy.add_transition(infected, source_data_type='rate')
        infected.add_transition(healthy)

        builder.components.add_components([DiseaseModel(self.cause, states=[healthy, infected],
                                                        get_data_functions=get_data_functions)])


class neonatal:

    def __init__(self, cause):
        self.cause = cause

    def setup(self, builder):

        only_morbid = builder.data.load(f'cause.{self.cause}.restrictions')['yld_only']

        healthy = SusceptibleState(self.cause)
        healthy.allow_self_transitions()

        get_data_functions = {}
        if only_morbid:
            with_condition = DiseaseState(self.cause)
            get_data_functions['csmr'] = lambda _, __: None
        else:
            with_condition = ExcessMortalityState(self.cause)
        with_condition.allow_self_transitions()

        # TODO: some neonatal causes (e.g. sepsis) have incidence and remission at least at the MEID level
        # healthy.add_transition(with_condition, source_data_type='rate')
        # with_condition.add_transition(healthy, source_data_type='rate')

        builder.components.add_components([DiseaseModel(self.cause, states=[healthy, with_condition],
                                                        get_data_functions=get_data_functions)])






