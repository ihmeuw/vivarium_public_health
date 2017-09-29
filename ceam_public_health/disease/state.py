"""A toolbox for modeling diseases as state machines."""
import numpy as np
import pandas as pd

from vivarium.framework.event import listens_for
from vivarium.framework.state_machine import State, Transient
from vivarium.framework.values import modifies_value

from ceam_public_health.disease import RateTransition, ProportionTransition

from ceam_inputs import get_disability_weight, get_prevalence, get_excess_mortality, get_duration, SeveritySplit


class BaseDiseaseState(State):
    def __init__(self, cause, side_effect_function=None, track_events=True, **kwargs):
        super().__init__(cause.name, **kwargs)
        self.cause = cause
        self.side_effect_function = side_effect_function

        self.track_events = track_events
        self.event_time_column = self.state_id + '_event_time'
        self.event_count_column = self.state_id + '_event_count'

        # Condition is set when the state is added to a disease model
        self.condition = None

    def setup(self, builder):
        """Performs this component's simulation setup and return sub-components.

        Parameters
        ----------
        builder : `engine.Builder`
            Interface to several simulation tools.

        Returns
        -------
        iterable
            This component's sub-components.
        """
        subcomponents = super().setup(builder)
        self.clock = builder.clock()

        columns = [self.condition, 'alive']
        if self.track_events:
            columns += [self.event_time_column, self.event_count_column]
        self.population_view = builder.population_view(columns)

        return super().setup(builder)

    def _transition_side_effect(self, index, event_time):
        """Updates the simulation state and triggers any side-effects associated with this state.

        Parameters
        ----------
        index : iterable of ints
            An iterable of integer labels for the simulants.
        event_time : pandas.Timestamp
            The time at which this transition occurs.
        """
        if self.track_events:
            pop = self.population_view.get(index)
            pop[self.event_time_column] = event_time
            pop[self.event_count_column] += 1
            self.population_view.update(pop)

        if self.side_effect_function is not None:
            self.side_effect_function(index, event_time)

    @listens_for('initialize_simulants')
    def load_population_columns(self, event):
        """Adds this state's columns to the simulation state table.

        Parameters
        ----------
        event : `vivarium.framework.population.PopulationEvent`
            An event signaling the creation of new simulants.
        """
        if self.track_events:
            self.population_view.update(pd.DataFrame({self.event_time_column: pd.Series(pd.NaT, index=event.index),
                                                      self.event_count_column: pd.Series(0, index=event.index)},
                                                     index=event.index))

        for transition in self.transition_set:
            if transition.start_active:
                transition.set_active(event.index)

    def add_transition(self, output, data_type=None, **kwargs):
        transition_map = {'rate': RateTransition, 'proportion': ProportionTransition}
        if not data_type:
            return super().add_transition(output, **kwargs)
        elif data_type in transition_map:
            t = transition_map[data_type](self, output, **kwargs)
            self.transition_set.append(t)
            return t
        else:
            raise ValueError(f"Unrecognized data type {data_type}")

    @modifies_value('metrics')
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
        if self.track_events:
            population = self.population_view.get(index)
            metrics[self.event_count_column] = population[self.event_count_column].sum()
        return metrics


class DiseaseState(BaseDiseaseState):
    """State representing a disease in a state machine model."""
    def __init__(self, cause, cleanup_function=None, **kwargs):
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
        track_events : bool, optional
        """
        super().__init__(cause, **kwargs)
        self.cleanup_function = cleanup_function

    def setup(self, builder):
        """Performs this component's simulation setup and return sub-components.

        Parameters
        ----------
        builder : `engine.Builder`
            Interface to several simulation tools.

        Returns
        -------
        iterable
            This component's sub-components.
        """
        disability_weight_data = get_disability_weight(self.cause, builder.configuration)
        self.prevalence_data = get_prevalence(self.cause, builder.configuration)
        self._dwell_time = get_duration(self.cause, builder.configuration)

        if disability_weight_data is not None:
            self._disability_weight = builder.lookup(disability_weight_data)
        else:
            self._disability_weight = lambda index: pd.Series(np.zeros(len(index), dtype=float), index=index)

        self.dwell_time = builder.value('{}.dwell_time'.format(self.state_id))

        if isinstance(self._dwell_time, pd.DataFrame) or self._dwell_time.days > 0:
            self.transition_set.allow_null_transition = True
            self.track_events = True

        if isinstance(self._dwell_time, pd.Timedelta):
            self._dwell_time = self._dwell_time.total_seconds() / (60*60*24)
        self.dwell_time.source = builder.lookup(self._dwell_time)

        return super().setup(builder)

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
        population = self.population_view.get(index)
        # TODO: There is an uncomfortable overlap between having a dwell time and tracking events.
        if self.track_events:
            state_exit_time = population[self.event_time_column] + pd.to_timedelta(self.dwell_time(index), unit='D')
            return population.loc[state_exit_time <= event_time].index
        else:
            return index

    def _cleanup_effect(self, index, event_time):
        if self.cleanup_function is not None:
            self.cleanup_function(index, event_time)

    @modifies_value('disability_weight')
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

        return self._disability_weight(population.index) * ((population[self.condition] == self.state_id)
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
    def __init__(self, state_id, excess_mortality_data, **kwargs):
        super().__init__(state_id, **kwargs)



    def setup(self, builder):
        """Performs this component's simulation setup and return sub-components.
        Parameters
        ----------
        builder : `engine.Builder`
            Interface to several simulation tools.

        Returns
        -------
        iterable
             This component's sub-components.
        """
        self.excess_mortality_data = get_excess_mortality(self.cause, builder.configuration)
        self._mortality = builder.rate('{}.excess_mortality'.format(self.state_id))
        if 'mortality.interpolate' in builder.configuration and not builder.configuration.mortality.interpolate:
            order = 0
        else:
            order = 1
        self._mortality.source = builder.lookup(self.excess_mortality_data, interpolation_order=order)
        return super().setup(builder)

    @modifies_value('mortality_rate')
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
                * (population[self.condition] == self.state_id))
        if isinstance(rates_df, pd.Series):
            rates_df = pd.DataFrame({rates_df.name: rates_df, self.state_id: rate})
        else:
            rates_df[self.state_id] = rate
        return rates_df

    def __str__(self):
        return 'ExcessMortalityState({})'.format(self.state_id)



