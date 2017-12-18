import pandas as pd

from vivarium.framework.state_machine import Transition
from vivarium.framework.util import rate_to_probability
from vivarium.framework.values import list_combiner, joint_value_post_processor

from ceam_inputs import get_incidence


class RateTransition(Transition):
    def __init__(self, input_state, output_state, get_data_functions=None, **kwargs):
        super().__init__(input_state, output_state, probability_func=self._probability, **kwargs)
        self._get_data_functions = get_data_functions if get_data_functions is not None else {}

    def setup(self, builder):
        get_incidence_function = self._get_data_functions.get('incidence', get_incidence)
        self._rate_data = get_incidence_function(self.output_state.cause, builder.configuration)
        self.base_incidence = builder.lookup(self._rate_data)

        self.effective_incidence = builder.rate('{}.incidence_rate'.format(self.output_state.state_id))
        self.effective_incidence.source = self.incidence_rates

        self.joint_paf = builder.value('{}.paf'.format(self.output_state.state_id),
                                       list_combiner, joint_value_post_processor)
        self.joint_paf.source = lambda index: [pd.Series(0, index=index)]

        return super().setup(builder)

    def _probability(self, index):
        return rate_to_probability(self.effective_incidence(index))

    def incidence_rates(self, index):
        base_rates = self.base_incidence(index)
        joint_mediated_paf = self.joint_paf(index)
        # risk-deleted incidence is calculated by taking incidence from GBD and multiplying it by (1 - Joint PAF)
        return pd.Series(base_rates.values * (1 - joint_mediated_paf.values), index=index)

    def __str__(self):
        return f'RateTransition(from={self.input_state.state_id}, to={self.output_state.state_id})'


class ProportionTransition(Transition):
    def __init__(self, input_state, output_state, get_data_functions=None, **kwargs):
        super().__init__(input_state, output_state, probability_func=self._probability, **kwargs)
        self._get_data_functions = get_data_functions if get_data_functions is not None else {}

    def setup(self, builder):
        get_proportion_func = self._get_data_functions.get('proportion', None)
        if get_proportion_func is None:
            raise ValueError('Must supply a proportion function')
        self._proportion_data = get_proportion_func(self.output_state.cause, builder.configuration)
        self.proportion = builder.lookup(self._proportion_data)
        return super().setup(builder)

    def _probability(self, index):
        return self.proportion(index)

    def label(self):
        return super().label

    def __str__(self):
        return f'ProportionTransition(from={self.input_state.state_id}, {self.output_state.state_id})'
