"""
===================
The Model Menagerie
===================

This module contains a collection of frequently used parameterizations of
disease models.

"""
import pandas as pd

from vivarium_public_health.disease import (SusceptibleState, ExcessMortalityState, RecoveredState,
                                            DiseaseState, DiseaseModel)


class SI:

    def __init__(self, cause: str):
        self.cause = cause

    @property
    def name(self):
        return f'SI.{self.cause}'

    def setup(self, builder):
        only_morbid = builder.data.load(f'cause.{self.cause}.restrictions')['yld_only']
        disease_model_data_functions = {}

        healthy = SusceptibleState(self.cause)
        if only_morbid:
            infected = DiseaseState(self.cause)
            disease_model_data_functions['csmr'] = lambda _, __: None
        else:
            infected = ExcessMortalityState(self.cause)

        healthy.allow_self_transitions()
        healthy.add_transition(infected, source_data_type='rate')
        infected.allow_self_transitions()

        builder.components.add_components([DiseaseModel(self.cause, states=[healthy, infected],
                                                        get_data_functions=disease_model_data_functions)])

    def __repr__(self):
        return f'SI({self.cause})'


class SIR:

    def __init__(self, cause: str):
        self.cause = cause

    @property
    def name(self):
        return f'SIR.{self.cause}'

    def setup(self, builder):
        only_morbid = builder.data.load(f'cause.{self.cause}.restrictions')['yld_only']
        disease_model_data_functions = {}

        healthy = SusceptibleState(self.cause)
        if only_morbid:
            infected = DiseaseState(self.cause)
            disease_model_data_functions['csmr'] = lambda _, __: None
        else:
            infected = ExcessMortalityState(self.cause)
        recovered = RecoveredState(self.cause)

        healthy.allow_self_transitions()
        healthy.add_transition(infected, source_data_type='rate')
        infected.allow_self_transitions()
        infected.add_transition(recovered, source_data_type='rate')
        recovered.allow_self_transitions()

        builder.components.add_components([DiseaseModel(self.cause, states=[healthy, infected, recovered],
                                                        get_data_functions=disease_model_data_functions)])

    def __repr__(self):
        return f'SIR({self.cause})'


class SIS:

    def __init__(self, cause: str):
        self.cause = cause

    @property
    def name(self):
        return f'SIS.{self.cause}'

    def setup(self, builder):
        only_morbid = builder.data.load(f'cause.{self.cause}.restrictions')['yld_only']
        disease_model_data_functions = {}

        healthy = SusceptibleState(self.cause)
        if only_morbid:
            infected = DiseaseState(self.cause)
            disease_model_data_functions['csmr'] = lambda _, __: None
        else:
            infected = ExcessMortalityState(self.cause)

        healthy.allow_self_transitions()
        healthy.add_transition(infected, source_data_type='rate')
        infected.allow_self_transitions()
        infected.add_transition(healthy, source_data_type='rate')

        builder.components.add_components([DiseaseModel(self.cause, states=[healthy, infected],
                                                        get_data_functions=disease_model_data_functions)])

    def __repr__(self):
        return f'SIS({self.cause})'


class SIS_fixed_duration:

    def __init__(self, cause: str, duration: str):
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

    @property
    def name(self):
        return f'SIS_fixed_duration.{self.cause}.{self.duration}'

    def setup(self, builder):
        only_morbid = builder.data.load(f'cause.{self.cause}.restrictions')['yld_only']
        disease_model_data_functions = {}

        healthy = SusceptibleState(self.cause)
        if only_morbid:
            infected = DiseaseState(self.cause,
                                    get_data_functions={'dwell_time': lambda _, __: self.duration})
            disease_model_data_functions['csmr'] = lambda _, __: None
        else:
            infected = ExcessMortalityState(self.cause,
                                            get_data_functions={'dwell_time': lambda _, __: self.duration})

        healthy.allow_self_transitions()
        healthy.add_transition(infected, source_data_type='rate')
        infected.add_transition(healthy)
        infected.allow_self_transitions()

        builder.components.add_components([DiseaseModel(self.cause, states=[healthy, infected],
                                                        get_data_functions=disease_model_data_functions)])

    def __repr__(self):
        return f'SIS_fixed_duration(cause={self.cause}, duration={self.duration})'


class SIR_fixed_duration:

    def __init__(self, cause: str, duration: str):
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

    @property
    def name(self):
        return f'SIR_fixed_duration.{self.cause}.{self.duration}'

    def setup(self, builder):
        only_morbid = builder.data.load(f'cause.{self.cause}.restrictions')['yld_only']
        disease_model_data_functions = {}

        healthy = SusceptibleState(self.cause)
        if only_morbid:
            infected = DiseaseState(self.cause,
                                    get_data_functions={'dwell_time': lambda _, __: self.duration})
            disease_model_data_functions['csmr'] = lambda _, __: None
        else:
            infected = ExcessMortalityState(self.cause,
                                            get_data_functions={'dwell_time': lambda _, __: self.duration})
        recovered = RecoveredState(self.cause)

        healthy.allow_self_transitions()
        healthy.add_transition(infected, source_data_type='rate')
        infected.allow_self_transitions()
        infected.add_transition(recovered)
        recovered.allow_self_transitions()

        builder.components.add_components([DiseaseModel(self.cause, states=[healthy, infected, recovered],
                                                        get_data_functions=disease_model_data_functions)])

    def __repr__(self):
        return f'SIR_fixed_duration(cause={self.cause}, duration={self.duration})'


class NeonatalSWC_without_incidence:

    def __init__(self, cause: str):
        self.cause = cause

    @property
    def name(self):
        return f'NeonatalSWC_without_incidence.{self.cause}'

    def setup(self, builder):
        only_morbid = builder.data.load(f'cause.{self.cause}.restrictions')['yld_only']
        disease_model_data_functions = {}

        healthy = SusceptibleState(self.cause)

        with_condition_data_functions = {'birth_prevalence':
                                         lambda cause, builder: builder.data.load(f"cause.{cause}.birth_prevalence")}
        if only_morbid:
            with_condition = DiseaseState(self.cause, get_data_functions=with_condition_data_functions)
            disease_model_data_functions['csmr'] = lambda _, __: None
        else:
            with_condition = ExcessMortalityState(self.cause, get_data_functions=with_condition_data_functions)

        healthy.allow_self_transitions()
        with_condition.allow_self_transitions()

        builder.components.add_components([DiseaseModel(self.cause, states=[healthy, with_condition],
                                                        get_data_functions=disease_model_data_functions)])

    def __repr__(self):
        return f'NeonatalSWC_without_incidence({self.cause})'


class NeonatalSWC_with_incidence:

    def __init__(self, cause: str):
        self.cause = cause

    @property
    def name(self):
        return f'NeonatalSWC_with_incidence.{self.cause}'

    def setup(self, builder):
        only_morbid = builder.data.load(f'cause.{self.cause}.restrictions')['yld_only']
        disease_model_data_functions = {}

        healthy = SusceptibleState(self.cause)

        with_condition_data_functions = {'birth_prevalence':
                                         lambda cause, builder: builder.data.load(f"cause.{cause}.birth_prevalence")}
        if only_morbid:
            with_condition = DiseaseState(self.cause, get_data_functions=with_condition_data_functions)
            disease_model_data_functions['csmr'] = lambda _, __: None
        else:
            with_condition = ExcessMortalityState(self.cause, get_data_functions=with_condition_data_functions)

        healthy.allow_self_transitions()
        healthy.add_transition(with_condition, source_data_type='rate')
        with_condition.allow_self_transitions()

        builder.components.add_components([DiseaseModel(self.cause, states=[healthy, with_condition],
                                                        get_data_functions=disease_model_data_functions)])

    def __repr__(self):
        return f'NeonatalSWC_with_incidence({self.cause})'
