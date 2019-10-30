"""
===================
Disease Transitions
===================

This module contains tools to model transitions between disease states.

"""
import pandas as pd

from vivarium.framework.state_machine import Transition
from vivarium.framework.utilities import rate_to_probability
from vivarium.framework.values import list_combiner, union_post_processor


class RateTransition(Transition):
    def __init__(self, input_state, output_state, get_data_functions=None, **kwargs):
        super().__init__(input_state, output_state, probability_func=self._probability, **kwargs)
        self._get_data_functions = get_data_functions if get_data_functions is not None else {}

    def setup(self, builder):
        rate_data, pipeline_name = self.load_transition_rate_data(builder)
        self.base_rate = builder.lookup.build_table(rate_data, key_columns=['sex'], parameter_columns=['age', 'year'])
        self.transition_rate = builder.value.register_rate_producer(pipeline_name,
                                                                    source=self.compute_transition_rate,
                                                                    requires_columns=['age', 'sex', 'alive'],
                                                                    requires_values=[f'{pipeline_name}.paf'])
        paf = builder.lookup.build_table(0)
        self.joint_paf = builder.value.register_value_producer(f'{pipeline_name}.paf',
                                                               source=lambda index: [paf(index)],
                                                               preferred_combiner=list_combiner,
                                                               preferred_post_processor=union_post_processor)

        self.population_view = builder.population.get_view(['alive'])

    def compute_transition_rate(self, index):
        transition_rate = pd.Series(0, index=index)
        living = self.population_view.get(index, query='alive == "alive"').index
        base_rates = self.base_rate(living)
        joint_paf = self.joint_paf(living)
        transition_rate.loc[living] = base_rates * (1 - joint_paf)
        return transition_rate

    def load_transition_rate_data(self, builder):
        if 'incidence_rate' in self._get_data_functions:
            rate_data = self._get_data_functions['incidence_rate'](self.output_state.cause, builder)
            pipeline_name = f'{self.output_state.state_id}.incidence_rate'
        elif 'remission_rate' in self._get_data_functions:
            rate_data = self._get_data_functions['remission_rate'](self.output_state.cause, builder)
            pipeline_name = f'{self.input_state.state_id}.remission_rate'
        else:
            raise ValueError("No valid data functions supplied.")
        return rate_data, pipeline_name

    def _probability(self, index):
        return rate_to_probability(self.transition_rate(index))

    def __str__(self):
        return f'RateTransition(from={self.input_state.state_id}, to={self.output_state.state_id})'


class ProportionTransition(Transition):
    def __init__(self, input_state, output_state, get_data_functions=None, **kwargs):
        super().__init__(input_state, output_state, probability_func=self._probability, **kwargs)
        self._get_data_functions = get_data_functions if get_data_functions is not None else {}

    def setup(self, builder):
        super().setup(builder)
        get_proportion_func = self._get_data_functions.get('proportion', None)
        if get_proportion_func is None:
            raise ValueError('Must supply a proportion function')
        self._proportion_data = get_proportion_func(self.output_state.cause, builder)
        self.proportion = builder.lookup.build_table(self._proportion_data, key_columns=['sex'],
                                                     parameter_columns=['age', 'year'])

    def _probability(self, index):
        return self.proportion(index)

    def __str__(self):
        return f'ProportionTransition(from={self.input_state.state_id}, {self.output_state.state_id})'
