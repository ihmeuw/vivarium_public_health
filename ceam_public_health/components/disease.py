import os.path
from datetime import timedelta
from functools import partial
import numbers

import pandas as pd
import numpy as np

from ceam import config

from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns
from ceam.framework.values import modifies_value, produces_value, list_combiner, joint_value_post_processor
from ceam.framework.util import rate_to_probability
from ceam.framework.state_machine import Machine, State, Transition, TransitionSet

from ceam_inputs import get_disease_states, get_proportion


class DiseaseState(State):
    def __init__(self, state_id, disability_weight, dwell_time=0,
                 event_time_column=None, event_count_column=None, side_effect_function=None,
                 track_events=False):
        State.__init__(self, state_id)

        self.side_effect_function = side_effect_function

        # Condition is set when the state is added to a disease model
        self.condition = None
        self._disability_weight = disability_weight
        self.dwell_time = dwell_time
        self.track_events = track_events or dwell_time > 0

        if isinstance(self.dwell_time, timedelta):
            self.dwell_time = self.dwell_time.total_seconds()

        if event_time_column:
            self.event_time_column = event_time_column
        else:
            self.event_time_column = self.state_id + '_event_time'

        if event_count_column:
            self.event_count_column = event_count_column
        else:
            self.event_count_column = self.state_id + '_event_count'

    def setup(self, builder):
        columns = [self.condition]
        if self.dwell_time > 0:
            columns += [self.event_time_column, self.event_count_column]
        self.population_view = builder.population_view(columns)
        self.clock = builder.clock()

    @listens_for('initialize_simulants')
    def load_population_columns(self, event):
        if self.track_events:
            population_size = len(event.index)
            self.population_view.update(pd.DataFrame({self.event_time_column: np.zeros(population_size),
                                                      self.event_count_column: np.zeros(population_size)},
                                                     index=event.index))

    def next_state(self, index, population_view):
        if self.dwell_time > 0:
            population = self.population_view.get(index)
            eligible_index = population.loc[population[self.event_time_column] < self.clock().timestamp() - self.dwell_time].index
        else:
            eligible_index = index
        return super(DiseaseState, self).next_state(eligible_index, population_view)

    def _transition_side_effect(self, index):
        if self.dwell_time > 0:
            pop = self.population_view.get(index)
            pop[self.event_time_column] = self.clock().timestamp()
            pop[self.event_count_column] += 1
            self.population_view.update(pop)
        if self.side_effect_function is not None:
            self.side_effect_function(index)

    @modifies_value('metrics')
    def metrics(self, index, metrics):
        if self.dwell_time > 0:
            population = self.population_view.get(index)
            metrics[self.event_count_column] = population[self.event_count_column].sum()
        return metrics

    @modifies_value('disability_weight')
    def disability_weight(self, index):
        population = self.population_view.get(index)
        return self._disability_weight * (population[self.condition] == self.state_id)


class ExcessMortalityState(DiseaseState):
    def __init__(self, state_id, excess_mortality_data, prevalence_data, csmr_data, **kwargs):
        DiseaseState.__init__(self, state_id, **kwargs)

        self.excess_mortality_data = excess_mortality_data
        self.prevalence_data = prevalence_data
        self.csmr_data = csmr_data

    def setup(self, builder):
        self.mortality = builder.rate('{}.excess_mortality'.format(self.state_id))
        self.mortality.source = builder.lookup(self.excess_mortality_data)
        return super(ExcessMortalityState, self).setup(builder)

    @modifies_value('mortality_rate')
    def mortality_rates(self, index, rates_df):
        population = self.population_view.get(index)
        rates_df[self.state_id] = self.mortality(population.index, skip_post_processor=True) * (population[self.condition] == self.state_id)

        return rates_df

    @modifies_value('csmr_data')
    def mmeids(self):
        return self.csmr_data

    def name(self):
        return '{}'.format(self.state_id)

    def __str__(self):
        return 'ExcessMortalityState("{}" ...)'.format(self.state_id)


class RateTransition(Transition):
    def __init__(self, output, rate_label, rate_data, name_prefix='incidence_rate'):
        Transition.__init__(self, output, self.probability)

        self.rate_label = rate_label
        self.rate_data = rate_data
        self.name_prefix = name_prefix

    def setup(self, builder):
        self.effective_incidence = builder.rate('{}.{}'.format(self.name_prefix, self.rate_label))
        self.effective_incidence.source = self.incidence_rates
        self.joint_paf = builder.value('paf.{}'.format(self.rate_label), list_combiner, joint_value_post_processor)
        self.joint_paf.source = lambda index: [pd.Series(0, index=index)]
        self.base_incidence = builder.lookup(self.rate_data)

    def probability(self, index):
        return rate_to_probability(self.effective_incidence(index))

    def incidence_rates(self, index):
        base_rates = self.base_incidence(index)
        joint_mediated_paf = self.joint_paf(index)

        # risk-deleted incidence is calculated by taking incidence from GBD and multiplying it by (1 - Joint PAF)
        return pd.Series(base_rates.values * (1 - joint_mediated_paf.values), index=index)

    def __str__(self):
        return 'RateTransition("{0}", "{1}")'.format(
            self.output.state_id if hasattr(self.output, 'state_id')
            else [str(x) for x in self.output], self.rate_label)


class ProportionTransition(Transition):
    def __init__(self, output, modelable_entity_id=None, proportion=None):
        Transition.__init__(self, output, self.probability)

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

    def probability(self, index):
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
        return 'ProportionTransition("{}", "{}", "{}")'.format(self.output.state_id if hasattr(self.output, 'state_id') else [str(x) for x in self.output], self.modelable_entity_id, self.proportion)



class DiseaseModel(Machine):
    def __init__(self, condition):
        Machine.__init__(self, condition)

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
    @uses_columns(['age', 'sex'])
    def load_population_columns(self, event):
        population = event.population

        state_map = {s.state_id:s.prevalence_data for s in self.states if hasattr(s, 'prevalence_data')}

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
                            cube = cube.append(pd.DataFrame({'measure': 'prevalence', 'age_low': low, 'age_high': high, 'sex': sex, 'location': location if location >= 0 else root_location, 'cause': cause, 'value': affected/len(sub_pop), 'sample_size': len(sub_pop)}, index=[0]).set_index(['measure', 'age_low', 'age_high', 'sex', 'location', 'cause']))
        return cube

    @modifies_value('metrics')
    def metrics(self, index, metrics):
        population = self.population_view.get(index)
        metrics[self.condition + '_count'] = (population[self.condition] != 'healthy').sum()
        return metrics
