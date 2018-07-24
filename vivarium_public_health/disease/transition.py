import pandas as pd

from vivarium.framework.state_machine import Transition
from vivarium.framework.utilities import rate_to_probability
from vivarium.framework.values import list_combiner, joint_value_post_processor


class RateTransition(Transition):
    def __init__(self, input_state, output_state, get_data_functions=None, **kwargs):
        super().__init__(input_state, output_state, probability_func=self._probability, **kwargs)
        self._get_data_functions = get_data_functions if get_data_functions is not None else {}

    def setup(self, builder):
        rate_data, pipeline_name = self._get_rate_data(builder)
        self.base_rate = builder.lookup.build_table(rate_data)
        self.effective_rate = builder.value.register_rate_producer(pipeline_name, source=self.rates)
        self.joint_paf = builder.value.register_value_producer(f'{self.output_state.state_id}.paf',
                                                               source=lambda index: [pd.Series(0, index=index)],
                                                               preferred_combiner=list_combiner,
                                                               preferred_post_processor=joint_value_post_processor)

    def _get_rate_data(self, builder):
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
        return rate_to_probability(self.effective_rate(index))

    def rates(self, index):
        base_rates = self.base_rate(index)
        joint_mediated_paf = self.joint_paf(index)
        # risk-deleted incidence is calculated by taking incidence and multiplying it by (1 - Joint PAF)
        return pd.Series(base_rates.values * (1 - joint_mediated_paf.values), index=index)

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
        self.proportion = builder.lookup.build_table(self._proportion_data)

    def _probability(self, index):
        return self.proportion(index)

    def __str__(self):
        return f'ProportionTransition(from={self.input_state.state_id}, {self.output_state.state_id})'
