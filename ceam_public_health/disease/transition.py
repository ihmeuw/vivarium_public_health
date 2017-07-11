import numbers

import pandas as pd

from ceam_inputs import get_proportion
from vivarium.framework.state_machine import Transition
from vivarium.framework.util import rate_to_probability
from vivarium.framework.values import list_combiner, joint_value_post_processor


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
    def __init__(self, output, modelable_entity_id=None, proportion=None, **kwargs):
        super().__init__(output, probability_func=self._probability, **kwargs)

        if modelable_entity_id and proportion:
            raise ValueError("Must supply modelable_entity_id or proportion (proportion can be an int or df) but not both")

        if modelable_entity_id is None and proportion is None:
            raise ValueError("Must supply either modelable_entity_id or proportion (proportion can be int or df)")

        self.modelable_entity_id = modelable_entity_id
        self.proportion = proportion

    def setup(self, builder):
        if self.modelable_entity_id:
            self.proportion = builder.lookup(get_proportion(self.modelable_entity_id))
        elif not isinstance(self.proportion, numbers.Number):
            self.proportion = builder.lookup(self.proportion)
        return super().setup(builder)

    def _probability(self, index):
        if callable(self.proportion):
            return self.proportion(index)
        else:
            return pd.Series(self.proportion, index=index)

    def label(self):
        if self.modelable_entity_id:
            return str(self.modelable_entity_id)
        elif isinstance(self.proportion, numbers.Number):
            return '{:.3f}'.format(self.proportion)
        else:
            return super().label()

    def __str__(self):
        return 'ProportionTransition({}, {}, {})'.format(self.output.state_id if hasattr(self.output, 'state_id') else [str(x) for x in self.output], self.modelable_entity_id, self.proportion)