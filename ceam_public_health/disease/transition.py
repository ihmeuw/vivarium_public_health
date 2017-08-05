import numbers

import pandas as pd

from vivarium.framework.state_machine import Transition
from vivarium.framework.util import rate_to_probability
from vivarium.framework.values import list_combiner, joint_value_post_processor


class RateTransition(Transition):
    def __init__(self, output, rate_label, rate_data, **kwargs):
        super().__init__(output, probability_func=self._probability, **kwargs)

        self.rate_label = rate_label
        self.rate_data = rate_data

    def setup(self, builder):
        self.effective_incidence = builder.rate('{}.incidence_rate'.format(self.rate_label))
        self.effective_incidence.source = self.incidence_rates
        self.joint_paf = builder.value('{}.paf'.format(self.rate_label), list_combiner, joint_value_post_processor)
        self.joint_paf.source = lambda index: [pd.Series(0, index=index)]
        self.base_incidence = builder.lookup(self.rate_data)
        return super().setup(builder)

    def _probability(self, index):
        return rate_to_probability(self.effective_incidence(index))

    def incidence_rates(self, index):
        base_rates = self.base_incidence(index)
        joint_mediated_paf = self.joint_paf(index)
        # risk-deleted incidence is calculated by taking incidence from GBD and multiplying it by (1 - Joint PAF)
        return pd.Series(base_rates.values * (1 - joint_mediated_paf.values), index=index)

    def __str__(self):
        return 'RateTransition({0}, {1})'.format(
            self.output.state_id if hasattr(self.output, 'state_id') else [str(x) for x in self.output],
            self.rate_label)


class ProportionTransition(Transition):
    def __init__(self, output, proportion, **kwargs):
        super().__init__(output, probability_func=self._probability, **kwargs)
        self.proportion = proportion

    def setup(self, builder):
        if not isinstance(self.proportion, numbers.Number):
            self.proportion = builder.lookup(self.proportion)
        return super().setup(builder)

    def _probability(self, index):
        if callable(self.proportion):
            return self.proportion(index)
        else:
            return pd.Series(self.proportion, index=index)

    def label(self):
        if isinstance(self.proportion, numbers.Number):
            return '{:.3f}'.format(self.proportion)
        else:
            return super().label()

    def __str__(self):
        return 'ProportionTransition({}, {})'.format(self.output.state_id if hasattr(self.output, 'state_id')
                                                     else [str(x) for x in self.output], self.proportion)
