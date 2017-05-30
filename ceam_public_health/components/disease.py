"""A toolbox for modeling diseases as state machines."""

from datetime import timedelta
import numbers

import pandas as pd
import numpy as np

from ceam import config

from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns
from ceam.framework.values import modifies_value, list_combiner, joint_value_post_processor
from ceam.framework.util import rate_to_probability
from ceam.framework.state_machine import Machine, State, TransientState, Transition, TransitionSet

from ceam_inputs import (get_disease_states, get_proportion, get_cause_specific_mortality,
                         get_disability_weight, get_prevalence, get_excess_mortality, meid, hid)


class DiseaseState(State):
    """State representing a disease in a state machine model.
    
    Parameters
    ----------
    state_id : str
        The name of this state.
    disability_weight : `pandas.DataFrame`, optional
        The amount of disability associated with this state.
    prevalence_data : `pandas.DataFrame`, optional
        The baseline occurrence of this state in a population.
    dwell_time : `pandas.DataFrame` or float or `datetime.timedelta`, optional
        The minimum time a simulant exists in this state.
    event_time_column : str, optional
        The name of a column to track the last time this state was entered.
    event_count_column : str, optional
        The name of a column to track the number of times this state was entered.
    side_effect_function : callable, optional
        A function to be called when this state is entered.
    track_events : bool, optional
    
    Attributes
    ----------
    state_id : str
        The name of this state.    
    prevalence_data : `pandas.DataFrame`
        The baseline occurrence of this state in a population.
    dwell_time : `pandas.DataFrame`
        The minimum time a simulant exists in this state.
    event_time_column : str
        The name of a column to track the last time this state was entered.
    event_count_column : str
        The name of a column to track the number of times this state was entered.
    side_effect_function : callable
        A function to be called when this state is entered.
    track_events : bool    
        Flag to indicate whether the last time this state was entered and the number of times 
        it has been entered should be tracked.
    condition : ?
    population_view : `pandas.DataFrame`
        A view into the simulation state table.
    """
    def __init__(self, state_id, disability_weight=None, prevalence_data=None,
                 dwell_time=0, event_time_column=None, event_count_column=None,
                 side_effect_function=None, track_events=True, key='state'):
        super().__init__(state_id, key=key)
        self._disability_weight_data = disability_weight
        self.prevalence_data = prevalence_data
        self._dwell_time = dwell_time
        if self._dwell_time is not None:
            self.transition_set.allow_null_transition = True
        self.event_time_column = event_time_column if event_time_column else self.state_id + '_event_time'
        self.event_count_column = event_count_column if event_count_column else self.state_id + '_event_count'
        self.side_effect_function = side_effect_function
        self.track_events = track_events or isinstance(self._dwell_time, pd.DataFrame) or self._dwell_time > 0
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
        columns = [self.condition]
        if self.track_events:
            columns += [self.event_time_column, self.event_count_column]
        self.population_view = builder.population_view(columns)
        self.clock = builder.clock()
        if self._disability_weight_data is not None:
            self._disability_weight = builder.lookup(self._disability_weight_data)
        else:
            self._disability_weight = lambda index: pd.Series(np.zeros(len(index), dtype=float), index=index)
        self.dwell_time = builder.value('dwell_time.{}'.format(self.state_id))
        if isinstance(self._dwell_time, timedelta):
            self._dwell_time = self._dwell_time.total_seconds() / (60*60*24)

        self.dwell_time.source = builder.lookup(self._dwell_time)
        return super().setup(builder)

    @listens_for('initialize_simulants')
    def load_population_columns(self, event):
        """Adds this state's columns to the simulation state table.
        
        Parameters
        ----------
        event : `ceam.framework.population.PopulationEvent`
            An event signaling the creation of new simulants.
        """
        if self.track_events:
            population_size = len(event.index)
            self.population_view.update(pd.DataFrame({self.event_time_column: pd.Series([pd.NaT]*population_size),
                                                      self.event_count_column: np.zeros(population_size)},
                                                     index=event.index))

    def next_state(self, index, population_view):
        """Moves a population among different disease states.    
        
        Parameters
        ----------
        index : iterable of ints
            An iterable of integer labels for the simulants.    
        population_view : `pandas.DataFrame`
            A view of the internal state of the simulation.
        """
        eligible_index = self._filter_for_transition_eligibility(index)
        return super().next_state(eligible_index, population_view)

    def _filter_for_transition_eligibility(self, index):
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
        if self.track_events:  # TODO: There is an uncomfortable overlap between having a dwell time and tracking events.
            return population.loc[population[self.event_time_column] + pd.to_timedelta(self.dwell_time(index), unit='D')
                                  < pd.Timestamp(self.clock())
                                  + pd.Timedelta(config.simulation_parameters.time_step, unit='D')].index
        else:
            return index

    def _transition_side_effect(self, index):
        """Updates the simulation state and triggers any side-effects associated with this state.

        Parameters
        ----------
        index : iterable of ints
            An iterable of integer labels for the simulants.    
        """
        if self.track_events:
            pop = self.population_view.get(index)
            pop[self.event_time_column] = pd.Timestamp(self.clock())
            pop[self.event_count_column] += 1
            self.population_view.update(pop)
        if self.side_effect_function is not None:
            self.side_effect_function(index)

    def add_transition(self, output, proportion=None, rates=None, triggered=False):

        if proportion is not None and rates is not None:
            raise ValueError("Both proportion and rate data provided.")
        if proportion is not None:
            t = ProportionTransition(output=output,
                                     proportion=proportion,
                                     triggered=triggered)
            self.transition_set.append(t)
            return t
        elif rates is not None:
            t = RateTransition(output=output,
                               rate_label=output.name(),
                               rate_data=rates,
                               triggered=triggered)
            self.transition_set.append(t)
            return t
        else:
            return super().add_transition(output, triggered=triggered)


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
        return self._disability_weight(index) * (population[self.condition] == self.state_id)

    def name(self):
        return '{}'.format(self.state_id)

    def __str__(self):
        return 'DiseaseState({})'.format(self.state_id)


class TransientDiseaseState(TransientState):
    def __init__(self, state_id, event_time_column=None, event_count_column=None,
                 side_effect_function=None, track_events=True, key='state'):
        super().__init__(state_id, key=key)
        self.event_time_column = event_time_column if event_time_column else self.state_id + '_event_time'
        self.event_count_column = event_count_column if event_count_column else self.state_id + '_event_count'

        self.side_effect_function = side_effect_function
        self.track_events = track_events
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
        columns = [self.condition]
        if self.track_events:
            columns += [self.event_time_column, self.event_count_column]
        self.population_view = builder.population_view(columns)
        self.clock = builder.clock()
        return super().setup(builder)

    def _transition_side_effect(self, index):
        """Updates the simulation state and triggers any side-effects associated with this state.

        Parameters
        ----------
        index : iterable of ints
            An iterable of integer labels for the simulants.    
        """
        if self.track_events:
            pop = self.population_view.get(index)
            pop[self.event_time_column] = pd.Timestamp(self.clock().timestamp())
            pop[self.event_count_column] += 1
            self.population_view.update(pop)
        if self.side_effect_function is not None:
            self.side_effect_function(index)

    @listens_for('initialize_simulants')
    def load_population_columns(self, event):
        """Adds this state's columns to the simulation state table.

        Parameters
        ----------
        event : `ceam.framework.population.PopulationEvent`
            An event signaling the creation of new simulants.
        """
        if self.track_events:
            population_size = len(event.index)
            self.population_view.update(pd.DataFrame({self.event_time_column: pd.Series([pd.NaT]*population_size),
                                                      self.event_count_column: np.zeros(population_size)},
                                                     index=event.index))

    def add_transition(self, output, proportion=None, rates=None, triggered=False):

        if proportion is not None and rates is not None:
            raise ValueError("Both proportion and rate data provided.")
        if proportion is not None:
            t = ProportionTransition(output=output,
                                     proportion=proportion,
                                     triggered=triggered)
        elif rates is not None:
            t = RateTransition(output=output,
                               rate_label=output.name(),
                               rate_data=rates,
                               triggered=triggered)
        else:
            t = super().add_transition(output, triggered=triggered)
        self.transition_set.append(t)
        return t


class ExcessMortalityState(DiseaseState):
    """State representing a disease with excess mortality in a state machine model.
    
    Attributes
    ----------
    state_id : str
        The name of this state.
    excess_mortality_data : `pandas.DataFrame`
        A table of excess mortality data associated with this state.
    csmr_data : `pandas.DataFrame`
        A table of excess mortality data associated with this state.
    """
    def __init__(self, state_id, excess_mortality_data, csmr_data, **kwargs):
        super().__init__(state_id, **kwargs)

        self.excess_mortality_data = excess_mortality_data
        self.csmr_data = csmr_data

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
        self._mortality = builder.rate('{}.excess_mortality'.format(self.state_id))
        self._mortality.source = builder.lookup(self.excess_mortality_data)
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
            rates_df = pd.DataFrame({'rate': rates_df, self.state_id: rate})
        else:
            rates_df[self.state_id] = rate
        return rates_df

    @modifies_value('csmr_data')
    def get_csmr(self):
        return self.csmr_data

    def name(self):
        return '{}'.format(self.state_id)

    def __str__(self):
        return 'ExcessMortalityState({})'.format(self.state_id)


class RateTransition(Transition):
    def __init__(self, output, rate_label, rate_data, name_prefix='incidence_rate', **kwargs):
        super().__init__(output, probability_func=self._probability, **kwargs)

        self.rate_label = rate_label
        self.rate_data = rate_data
        self.name_prefix = name_prefix

    def setup(self, builder):
        self.effective_incidence = builder.rate('{}.{}'.format(self.name_prefix, self.rate_label))
        self.effective_incidence.source = self.incidence_rates
        self.joint_paf = builder.value('paf.{}'.format(self.rate_label), list_combiner, joint_value_post_processor)
        self.joint_paf.source = lambda index: [pd.Series(0, index=index)]
        self.base_incidence = builder.lookup(self.rate_data)

    def _probability(self, index):
        return rate_to_probability(self.effective_incidence(index))

    def incidence_rates(self, index):
        base_rates = self.base_incidence(index)
        joint_mediated_paf = self.joint_paf(index)
        # risk-deleted incidence is calculated by taking incidence from GBD and multiplying it by (1 - Joint PAF)
        return pd.Series(base_rates.values * (1 - joint_mediated_paf.values), index=index)

    def __str__(self):
        return 'RateTransition({0}, {1})'.format(
            self.output.state_id if hasattr(self.output, 'state_id')
            else [str(x) for x in self.output], self.rate_label)


class ProportionTransition(Transition):
    def __init__(self, output, modelable_entity_id=None, proportion=None, **kwargs):
        super().__init__(output, probability_func=self._probability, **kwargs)

        if modelable_entity_id and proportion:
            raise ValueError("Must supply modelable_entity_id or proportion (proportion can be an int or df) but not both")

        # @alecwd: had to change line below since it was erroring out when proportion is a dataframe. might be a cleaner way to do this that I don't know of
        if modelable_entity_id is None and proportion is None:
            raise ValueError("Must supply either modelable_entity_id or proportion (proportion can be int or df)")

        self.modelable_entity_id = modelable_entity_id
        self.proportion = proportion

    def setup(self, builder):
        if self.modelable_entity_id:
            self.proportion = builder.lookup(get_proportion(self.modelable_entity_id))
        elif not isinstance(self.proportion, numbers.Number):
            self.proportion = builder.lookup(self.proportion)

    def _probability(self, index):
        if callable(self.proportion):
            return self.proportion(index)
        else:
            return pd.Series(self.proportion, index=index)

    def label(self):
        if self.modelable_entity_id:
            return str(self.modelable_entity_id)
        else:
            return str(self.proportion)

    def __str__(self):
        return 'ProportionTransition({}, {}, {})'.format(self.output.state_id if hasattr(self.output, 'state_id') else [str(x) for x in self.output], self.modelable_entity_id, self.proportion)


class DiseaseModel(Machine):
    def __init__(self, condition, **kwargs):
        super().__init__(condition, **kwargs)

    def module_id(self):
        return str((self.__class__, self.state_column))

    @property
    def condition(self):
        return self.state_column

    def setup(self, builder):
        self.population_view = builder.population_view([self.condition], 'alive')

        sub_components = set()
        for state in self.states:
            state.condition = self.condition
            sub_components.add(state)
            sub_components.add(state.transition_set)
            for transition in state.transition_set:
                sub_components.add(transition)
                if isinstance(transition.output, TransitionSet):
                    sub_components.add(transition.output)
        return sub_components

    @listens_for('time_step')
    def time_step_handler(self, event):
        self.transition(event.index)


    @listens_for('initialize_simulants')
    @uses_columns(['age', 'sex', condition])
    def load_population_columns(self, event):
        population = event.population

        state_map = {s.state_id: s.prevalence_data for s in self.states
                     if hasattr(s, 'prevalence_data') and s.prevalence_data is not None}

        if state_map:
            # only do this if there are states in the model that supply prevalence data
            population['sex_id'] = population.sex.apply({'Male':1, 'Female':2}.get)
            condition_column = get_disease_states(population, state_map)
            condition_column = condition_column.rename(columns={'condition_state': self.condition})
        else:
            condition_column = pd.Series('healthy', index=population.index, name=self.condition)
        self.population_view.update(condition_column)

    @modifies_value('epidemiological_point_measures')
    def prevalence(self, index, age_groups, sexes, all_locations, duration, cube):
        root_location = config.simulation_parameters.location_id
        pop = self.population_view.manager.population.ix[index].query('alive')
        causes = set(pop[self.condition]) - {'healthy'}
        if all_locations:
            locations = set(pop.location) | {-1}
        else:
            locations = {-1}
        for low, high in age_groups:
            for sex in sexes:
                for cause in causes:
                    for location in locations:
                        sub_pop = pop.query('age > @low and age <= @high and sex == @sex')
                        if location >= 0:
                            sub_pop = sub_pop.query('location == @location')
                        if not sub_pop.empty:
                            affected = (sub_pop[self.condition] == cause).sum()
                            cube = cube.append(pd.DataFrame({'measure': 'prevalence',
                                                             'age_low': low,
                                                             'age_high': high,
                                                             'sex': sex,
                                                             'location': location if location >= 0 else root_location,
                                                             'cause': cause, 'value': affected/len(sub_pop),
                                                             'sample_size': len(sub_pop)},
                                                            index=[0]).set_index(
                                ['measure', 'age_low', 'age_high', 'sex', 'location', 'cause']))
        return cube

    @modifies_value('metrics')
    def metrics(self, index, metrics):
        population = self.population_view.get(index)
        metrics[self.condition + '_count'] = (population[self.condition] != 'healthy').sum()
        return metrics


def make_disease_state(cause, dwell_time=0, side_effect_function=None):
    if 'mortality' in cause:
        csmr = get_cause_specific_mortality(cause.mortality) if isinstance(cause.mortality, meid) else cause.mortality
    else:
        csmr = pd.DataFrame()
    if 'disability_weight' in cause:
        if isinstance(cause.disability_weight, meid):
            disability_weight = get_disability_weight(dis_weight_modelable_entity_id=cause.disability_weight)
        elif isinstance(cause.disability_weight, hid):
            disability_weight = get_disability_weight(healthstate_id=cause.disability_weight)
        else:
            disability_weight = cause.disability_weight
    else:
        disability_weight = 0.0
    if 'prevalence' in cause:
        if isinstance(cause.prevalence, meid):
            prevalence = get_prevalence(cause.prevalence)
        else:
            prevalence = cause.prevalence
    else:
        prevalence = 0.0

    if 'excess_mortality' in cause:
        if isinstance(cause.excess_mortality, meid):
            excess_mortality = get_excess_mortality(cause.excess_mortality)
        else:
            excess_mortality = cause.excess_mortality
        return ExcessMortalityState(cause.name,
                                    dwell_time=dwell_time,
                                    disability_weight=disability_weight,
                                    excess_mortality_data=excess_mortality,
                                    prevalence_data=prevalence,
                                    csmr_data=csmr,
                                    side_effect_function=side_effect_function)
    else:
        return DiseaseState(cause.name,
                            dwell_time=dwell_time,
                            disability_weight=disability_weight,
                            prevalence_data=prevalence,
                            side_effect_function=side_effect_function)
