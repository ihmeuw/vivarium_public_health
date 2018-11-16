"""A toolbox for modeling diseases as state machines."""
import pandas as pd
import numpy as np

from vivarium.framework.state_machine import State, Transient

from vivarium_public_health.disease import RateTransition, ProportionTransition


class BaseDiseaseState(State):
    def __init__(self, cause, name_prefix=None, side_effect_function=None, cause_type="cause", **kwargs):
        self.cause = cause
        self.cause_type = cause_type
        cause_name = name_prefix + cause if name_prefix else cause
        super().__init__(cause_name, **kwargs)

        self.side_effect_function = side_effect_function

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
        if self.side_effect_function is not None:
            builder.components.add_components([self.side_effect_function])

        self.clock = builder.time.clock()

        columns = [self._model, 'alive']
        columns += [self.event_time_column, self.event_count_column]

        self.population_view = builder.population.get_view(columns)
        builder.population.initializes_simulants(self.load_population_columns,
                                                 creates_columns=[self.event_time_column, self.event_count_column],
                                                 requires_columns=[self._model])

        builder.value.register_value_modifier('metrics', self.metrics)

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

    def load_population_columns(self, pop_data):
        """Adds this state's columns to the simulation state table.

        Parameters
        ----------
        event : `vivarium.framework.population.PopulationEvent`
            An event signaling the creation of new simulants.
        """

        self.population_view.update(pd.DataFrame({self.event_time_column: pd.Series(pd.NaT, index=pop_data.index),
                                                  self.event_count_column: pd.Series(0, index=pop_data.index)},
                                                 index=pop_data.index))

        for transition in self.transition_set:
            if transition.start_active:
                transition.set_active(pop_data.index)

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

    def metrics(self, index, metrics):
        """Records data for simulation post-processing.

        Parameters
        ----------
        index : iterable of ints
            An iterable of integer labels for the simulants.
        metrics : `pandas.DataFrame`
            A table for recording simulation events of interest in post-processing.

        Returns
        -------
        `pandas.DataFrame`
            The metrics table updated to reflect new simulation state."""

        population = self.population_view.get(index)
        metrics[self.event_count_column] = population[self.event_count_column].sum()
        return metrics


class SusceptibleState(BaseDiseaseState):
    def __init__(self, cause, *args, **kwargs):
        super().__init__(cause, *args, name_prefix='susceptible_to_', **kwargs)

    def add_transition(self, output, source_data_type=None, get_data_functions=None, **kwargs):
        if source_data_type == 'rate':
            if get_data_functions is None:
                get_data_functions = {
                    'incidence_rate': lambda cause, builder: builder.data.load(f"{self.cause_type}.{cause}.incidence")
                }
            elif 'incidence_rate' not in get_data_functions:
                raise ValueError('You must supply an incidence rate function.')
        elif source_data_type == 'proportion':
            if 'proportion' not in get_data_functions:
                raise ValueError('You must supply a proportion function.')

        return super().add_transition(output, source_data_type, get_data_functions, **kwargs)


class RecoveredState(BaseDiseaseState):
    def __init__(self, cause, *args, **kwargs):
        super().__init__(cause, *args, name_prefix='recovered_from_', **kwargs)

    def add_transition(self, output, source_data_type=None, get_data_functions=None, **kwargs):
        if source_data_type == 'rate':
            if get_data_functions is None:
                get_data_functions = {
                    'incidence_rate': lambda cause, builder: builder.data.load(f"{self.cause_type}.{cause}.incidence")
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
        state_id : str
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
        get_disability_weight_func = self._get_data_functions.get(
            'disability_weight', lambda cause, builder: builder.data.load(
                f"{self.cause_type}.{cause}.disability_weight"))
        get_prevalence_func = self._get_data_functions.get(
            'prevalence', lambda cause, builder: builder.data.load(f"{self.cause_type}.{cause}.prevalence"))
        get_dwell_time_func = self._get_data_functions.get('dwell_time', lambda *args, **kwargs: pd.Timedelta(0))

        disability_weight_data = get_disability_weight_func(self.cause, builder)
        self.prevalence_data = builder.lookup.build_table(get_prevalence_func(self.cause, builder))
        self._dwell_time = get_dwell_time_func(self.cause, builder)
        self.randomness_prevalence = builder.randomness.get_stream(f'{self.state_id}_prevalent_cases')

        if isinstance(disability_weight_data, pd.DataFrame):
            self._disability_weight = builder.lookup.build_table(float(disability_weight_data.value))
        elif disability_weight_data is not None:
            self._disability_weight = builder.lookup.build_table(disability_weight_data)
        else:
            self._disability_weight = builder.lookup.build_table(0)
        builder.value.register_value_modifier('disability_weight', modifier=self.disability_weight)

        if isinstance(self._dwell_time, pd.DataFrame) or self._dwell_time.days > 0:
            self.transition_set.allow_null_transition = True

        if isinstance(self._dwell_time, pd.Timedelta):
            self._dwell_time = self._dwell_time.total_seconds() / (60*60*24)

        self.dwell_time = builder.value.register_value_producer(f'{self.state_id}.dwell_time',
                                                                source=builder.lookup.build_table(self._dwell_time))

    def load_population_columns(self, pop_data):
        super().load_population_columns(pop_data)
        simulants_with_condition = self.population_view.get(pop_data.index, query=f'{self._model}=="{self.state_id}"')
        if not simulants_with_condition.empty:
            infected_at = self._assign_event_time_for_prevalent_cases(simulants_with_condition, self.clock(),
                                                                      self.randomness_prevalence.get_draw,
                                                                      self.dwell_time)
            infected_at.name = self.event_time_column
            self.population_view.update(infected_at)

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
                    'remission_rate': lambda cause, builder: builder.data.load(f"{self.cause_type}.{cause}.remission")
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

    def disability_weight(self, index):
        """Gets the disability weight associated with this state.

        Parameters
        ----------
        index : iterable of ints
            An iterable of integer labels for the simulants.

        Returns
        -------
        `pandas.Series`
            An iterable of disability weights indexed by the provided `index`."""
        population = self.population_view.get(index)

        return self._disability_weight(population.index) * ((population[self._model] == self.state_id)
                                                            & (population.alive == 'alive'))

    def name(self):
        return '{}'.format(self.state_id)

    def __repr__(self):
        return 'DiseaseState({})'.format(self.state_id)


class TransientDiseaseState(BaseDiseaseState, Transient):

    def __repr__(self):
        return 'TransientDiseaseState(name={})'.format(self.state_id)


class ExcessMortalityState(DiseaseState):
    """State representing a disease with excess mortality in a state machine model.

    Attributes
    ----------
    state_id : str
        The name of this state.
    excess_mortality_data : `pandas.DataFrame`
        A table of excess mortality data associated with this state.
    """
    def __init__(self, cause, **kwargs):
        super().__init__(cause, **kwargs)

    def setup(self, builder):
        """Performs this component's simulation setup.
        Parameters
        ----------
        builder : `engine.Builder`
            Interface to several simulation tools.
        """
        super().setup(builder)
        get_excess_mortality_func = self._get_data_functions.get('excess_mortality', lambda cause, builder: builder.data.load(f"{self.cause_type}.{cause}.excess_mortality"))

        self.excess_mortality_data = get_excess_mortality_func(self.cause, builder)
        excess_mortality_source = builder.lookup.build_table(self.excess_mortality_data)
        self._mortality = builder.value.register_rate_producer(f'{self.state_id}.excess_mortality',
                                                               source=excess_mortality_source)
        builder.value.register_value_modifier('mortality_rate', modifier=self.mortality_rates)

    def mortality_rates(self, index, rates_df):
        """Modifies the baseline mortality rate for a simulant if they are in this state.

        Parameters
        ----------
        index : iterable of ints
            An iterable of integer labels for the simulants.
        rates_df : `pandas.DataFrame`

        """
        population = self.population_view.get(index)
        rate = (self._mortality(population.index, skip_post_processor=True)
                * (population[self._model] == self.state_id))
        if isinstance(rates_df, pd.Series):
            rates_df = pd.DataFrame({rates_df.name: rates_df, self.state_id: rate})
        else:
            rates_df[self.state_id] = rate
        return rates_df

    def __str__(self):
        return 'ExcessMortalityState({})'.format(self.state_id)
