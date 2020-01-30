"""
==============
Disease States
==============

This module contains tools to manage standard disease states.

"""
import pandas as pd
import numpy as np
from vivarium.framework.state_machine import State, Transient
from vivarium.framework.values import list_combiner, union_post_processor

from vivarium_public_health.disease import RateTransition, ProportionTransition


class BaseDiseaseState(State):
    def __init__(self, cause, name_prefix='', side_effect_function=None, cause_type="cause", **kwargs):
        super().__init__(name_prefix + cause)  # becomes state_id
        self.cause_type = cause_type
        self.cause = cause

        self.side_effect_function = side_effect_function
        if self.side_effect_function is not None:
            self._sub_components.append(side_effect_function)

        self.event_time_column = self.state_id + '_event_time'
        self.event_count_column = self.state_id + '_event_count'

    def setup(self, builder):
        """Performs this component's simulation setup.

        Parameters
        ----------
        builder : `engine.Builder`
            Interface to several simulation tools.
        """
        super().setup(builder)

        self.clock = builder.time.clock()

        columns_created = [self.event_time_column, self.event_count_column]
        view_columns = columns_created + [self._model, 'alive']
        self.population_view = builder.population.get_view(view_columns)
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 creates_columns=columns_created,
                                                 requires_columns=[self._model])

    def on_initialize_simulants(self, pop_data):
        """Adds this state's columns to the simulation state table."""
        for transition in self.transition_set:
            if transition.start_active:
                transition.set_active(pop_data.index)

        pop_update = pd.DataFrame({self.event_time_column: pd.NaT,
                                   self.event_count_column: 0},
                                  index=pop_data.index)
        self.population_view.update(pop_update)

    def _transition_side_effect(self, index, event_time):
        """Updates the simulation state and triggers any side-effects associated with this state.

        Parameters
        ----------
        index : iterable of ints
            An iterable of integer labels for the simulants.
        event_time : pandas.Timestamp
            The time at which this transition occurs.

        """
        pop = self.population_view.get(index)
        pop[self.event_time_column] = event_time
        pop[self.event_count_column] += 1
        self.population_view.update(pop)

        if self.side_effect_function is not None:
            self.side_effect_function(index, event_time)

    def add_transition(self, output, source_data_type=None, get_data_functions=None, **kwargs):
        transition_map = {'rate': RateTransition, 'proportion': ProportionTransition}

        if source_data_type is not None and source_data_type not in transition_map:
            raise ValueError(f"Unrecognized data type {source_data_type}")

        if not source_data_type:
            return super().add_transition(output, **kwargs)
        elif source_data_type in transition_map:
            t = transition_map[source_data_type](self, output, get_data_functions, **kwargs)
            self.transition_set.append(t)
            return t


class SusceptibleState(BaseDiseaseState):

    def __init__(self, cause, *args, **kwargs):
        super().__init__(cause, *args, name_prefix='susceptible_to_', **kwargs)

    def add_transition(self, output, source_data_type=None, get_data_functions=None, **kwargs):
        if source_data_type == 'rate':
            if get_data_functions is None:
                get_data_functions = {
                    'incidence_rate': lambda cause, builder: builder.data.load(f"{self.cause_type}.{cause}.incidence_rate")
                }
            elif 'incidence_rate' not in get_data_functions:
                raise ValueError('You must supply an incidence rate function.')
        elif source_data_type == 'proportion':
            if 'proportion' not in get_data_functions:
                raise ValueError('You must supply a proportion function.')

        return super().add_transition(output, source_data_type, get_data_functions, **kwargs)


class RecoveredState(BaseDiseaseState):

    def __init__(self, cause, *args, **kwargs):
        super().__init__(cause, *args, name_prefix="recovered_from_", **kwargs)

    def add_transition(self, output, source_data_type=None, get_data_functions=None, **kwargs):
        if source_data_type == 'rate':
            if get_data_functions is None:
                get_data_functions = {
                    'incidence_rate': lambda cause, builder: builder.data.load(f"{self.cause_type}.{cause}.incidence_rate")
                }
            elif 'incidence_rate' not in get_data_functions:
                raise ValueError('You must supply an incidence rate function.')
        elif source_data_type == 'proportion':
            if 'proportion' not in get_data_functions:
                raise ValueError('You must supply a proportion function.')

        return super().add_transition(output, source_data_type, get_data_functions, **kwargs)


class DiseaseState(BaseDiseaseState):
    """State representing a disease in a state machine model."""
    def __init__(self, cause, get_data_functions=None, cleanup_function=None, **kwargs):
        """
        Parameters
        ----------
        cause : str
            The name of this state.
        disability_weight : pandas.DataFrame or float, optional
            The amount of disability associated with this state.
        prevalence_data : pandas.DataFrame, optional
            The baseline occurrence of this state in a population.
        dwell_time : pandas.DataFrame or pandas.Timedelta, optional
            The minimum time a simulant exists in this state.
        event_time_column : str, optional
            The name of a column to track the last time this state was entered.
        event_count_column : str, optional
            The name of a column to track the number of times this state was entered.
        side_effect_function : callable, optional
            A function to be called when this state is entered.
        """
        super().__init__(cause, **kwargs)
        self._get_data_functions = get_data_functions if get_data_functions is not None else {}
        self.cleanup_function = cleanup_function

        if (self.cause is None and
                not set(self._get_data_functions.keys()).issuperset(['disability_weight', 'dwell_time', 'prevalence'])):
            raise ValueError('If you do not provide a cause, you must supply'
                             'custom data gathering functions for disability_weight, prevalence, and dwell_time.')

    def setup(self, builder):
        """Performs this component's simulation setup.

        Parameters
        ----------
        builder : `engine.Builder`
            Interface to several simulation tools.
        """
        super().setup(builder)

        prevalence_data = self.load_prevalence_data(builder)
        self.prevalence = builder.lookup.build_table(prevalence_data, key_columns=['sex'],
                                                     parameter_columns=['age', 'year'])

        birth_prevalence_data = self.load_birth_prevalence_data(builder)
        self.birth_prevalence = builder.lookup.build_table(birth_prevalence_data,
                                                           key_columns=['sex'],
                                                           parameter_columns=['year'])

        dwell_time_data = self.load_dwell_time_data(builder)
        self.dwell_time = builder.value.register_value_producer(f'{self.state_id}.dwell_time',
                                                                source=builder.lookup.build_table(dwell_time_data,
                                                                                                  key_columns=['sex'],
                                                                                                  parameter_columns=
                                                                                                  ['age', 'year']),
                                                                requires_columns=['age', 'sex'])

        disability_weight_data = self.load_disability_weight_data(builder)
        self.base_disability_weight = builder.lookup.build_table(disability_weight_data, key_columns=['sex'],
                                                                 parameter_columns=['age', 'year'])
        self.disability_weight = builder.value.register_value_producer(
            f'{self.state_id}.disability_weight',
            source=self.compute_disability_weight,
            requires_columns=['age', 'sex', 'alive', self._model]
        )
        builder.value.register_value_modifier('disability_weight', modifier=self.disability_weight)

        excess_mortality_data = self.load_excess_mortality_rate_data(builder)
        self.base_excess_mortality_rate = builder.lookup.build_table(excess_mortality_data, key_columns=['sex'],
                                                                     parameter_columns=['age', 'year'])
        self.excess_mortality_rate = builder.value.register_rate_producer(
            f'{self.state_id}.excess_mortality_rate',
            source=self.compute_excess_mortality_rate,
            requires_columns=['age', 'sex', 'alive', self._model],
            requires_values=[f'{self.state_id}.excess_mortality_rate.population_attributable_fraction']
        )
        paf = builder.lookup.build_table(0)
        self.joint_paf = builder.value.register_value_producer(
            f'{self.state_id}.excess_mortality_rate.population_attributable_fraction',
            source=lambda idx: [paf(idx)],
            preferred_combiner=list_combiner,
            preferred_post_processor=union_post_processor
        )
        builder.value.register_value_modifier('mortality_rate',
                                              modifier=self.adjust_mortality_rate,
                                              requires_values=[f'{self.state_id}.excess_mortality_rate'])

        self.randomness_prevalence = builder.randomness.get_stream(f'{self.state_id}_prevalent_cases')

    def on_initialize_simulants(self, pop_data):
        super().on_initialize_simulants(pop_data)
        simulants_with_condition = self.population_view.get(pop_data.index, query=f'{self._model}=="{self.state_id}"')
        if not simulants_with_condition.empty:
            infected_at = self._assign_event_time_for_prevalent_cases(simulants_with_condition, self.clock(),
                                                                      self.randomness_prevalence.get_draw,
                                                                      self.dwell_time)
            infected_at.name = self.event_time_column
            self.population_view.update(infected_at)

    def compute_disability_weight(self, index):
        """Gets the disability weight associated with this state.

        Parameters
        ----------
        index : iterable of ints
            An iterable of integer labels for the simulants.

        Returns
        -------
        `pandas.Series`
            An iterable of disability weights indexed by the provided `index`.
        """
        disability_weight = pd.Series(0, index=index)
        with_condition = self.with_condition(index)
        disability_weight.loc[with_condition] = self.base_disability_weight(with_condition)
        return disability_weight

    def compute_excess_mortality_rate(self, index):
        excess_mortality_rate = pd.Series(0, index=index)
        with_condition = self.with_condition(index)
        base_excess_mort = self.base_excess_mortality_rate(with_condition)
        joint_mediated_paf = self.joint_paf(with_condition)
        excess_mortality_rate.loc[with_condition] = base_excess_mort * (1 - joint_mediated_paf.values)
        return excess_mortality_rate

    def adjust_mortality_rate(self, index, rates_df):
        """Modifies the baseline mortality rate for a simulant if they are in this state.

        Parameters
        ----------
        index : iterable of ints
            An iterable of integer labels for the simulants.
        rates_df : `pandas.DataFrame`

        """
        rate = self.excess_mortality_rate(index, skip_post_processor=True)
        rates_df[self.state_id] = rate
        return rates_df

    def with_condition(self, index):
        pop = self.population_view.subview(['alive', self._model]).get(index)
        with_condition = pop.loc[(pop[self._model] == self.state_id) & (pop['alive'] == 'alive')].index
        return with_condition

    @staticmethod
    def _assign_event_time_for_prevalent_cases(infected, current_time, randomness_func, dwell_time_func):
        dwell_time = dwell_time_func(infected.index)
        infected_at = dwell_time * randomness_func(infected.index)
        infected_at = current_time - pd.to_timedelta(infected_at, unit='D')
        return infected_at

    def add_transition(self, output, source_data_type=None, get_data_functions=None, **kwargs):
        if source_data_type == 'rate':
            if get_data_functions is None:
                get_data_functions = {
                    'remission_rate': lambda cause, builder: builder.data.load(f"{self.cause_type}.{cause}.remission_rate")
                }
            elif 'remission_rate' not in get_data_functions:
                raise ValueError('You must supply a remission rate function.')
        elif source_data_type == 'proportion':
            if 'proportion' not in get_data_functions:
                raise ValueError('You must supply a proportion function.')
        return super().add_transition(output, source_data_type, get_data_functions, **kwargs)

    def next_state(self, index, event_time, population_view):
        """Moves a population among different disease states.

        Parameters
        ----------
        index : iterable of ints
            An iterable of integer labels for the simulants.
        event_time : pandas.Timestamp
            The time at which this transition occurs.
        population_view : vivarium.framework.population.PopulationView
            A view of the internal state of the simulation.
        """
        eligible_index = self._filter_for_transition_eligibility(index, event_time)
        return super().next_state(eligible_index, event_time, population_view)

    def _filter_for_transition_eligibility(self, index, event_time):
        """Filter out all simulants who haven't been in the state for the prescribed dwell time.

        Parameters
        ----------
        index : iterable of ints
            An iterable of integer labels for the simulants.

        Returns
        -------
        iterable of ints
            A filtered index of the simulants.
        """
        population = self.population_view.get(index, query='alive == "alive"')
        if np.any(self.dwell_time(index)) > 0:
            state_exit_time = population[self.event_time_column] + pd.to_timedelta(self.dwell_time(index), unit='D')
            return population.loc[state_exit_time <= event_time].index
        else:
            return index

    def _cleanup_effect(self, index, event_time):
        if self.cleanup_function is not None:
            self.cleanup_function(index, event_time)

    def load_prevalence_data(self, builder):
        if 'prevalence' in self._get_data_functions:
            return self._get_data_functions['prevalence'](self.cause, builder)
        else:
            return builder.data.load(f"{self.cause_type}.{self.cause}.prevalence")

    def load_birth_prevalence_data(self, builder):
        if 'birth_prevalence' in self._get_data_functions:
            return self._get_data_functions['birth_prevalence'](self.cause, builder)
        else:
            return 0

    def load_dwell_time_data(self, builder):
        if 'dwell_time' in self._get_data_functions:
            dwell_time = self._get_data_functions['dwell_time'](self.cause, builder)
        else:
            dwell_time = 0

        if isinstance(dwell_time, pd.Timedelta):
            dwell_time = dwell_time.total_seconds() / (60 * 60 * 24)
        if (isinstance(dwell_time, pd.DataFrame) and np.any(dwell_time.value != 0)) or dwell_time > 0:
            self.transition_set.allow_null_transition = True

        return dwell_time

    def load_disability_weight_data(self, builder):
        if 'disability_weight' in self._get_data_functions:
            disability_weight = self._get_data_functions['disability_weight'](self.cause, builder)
        else:
            disability_weight = builder.data.load(f"{self.cause_type}.{self.cause}.disability_weight")

        if isinstance(disability_weight, pd.DataFrame) and len(disability_weight) == 1:
            disability_weight = disability_weight.value[0]  # sequela only have single value

        return disability_weight

    def load_excess_mortality_rate_data(self, builder):
        only_morbid = builder.data.load(f'cause.{self._model}.restrictions')['yld_only']
        if 'excess_mortality_rate' in self._get_data_functions:
            return self._get_data_functions['excess_mortality_rate'](self.cause, builder)
        elif only_morbid:
            return 0
        else:
            return builder.data.load(f'{self.cause_type}.{self.cause}.excess_mortality_rate')

    def __repr__(self):
        return 'DiseaseState({})'.format(self.state_id)


class TransientDiseaseState(BaseDiseaseState, Transient):

    def __repr__(self):
        return 'TransientDiseaseState(name={})'.format(self.state_id)
