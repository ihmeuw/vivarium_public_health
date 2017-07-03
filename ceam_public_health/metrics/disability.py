import pandas as pd

from ceam.framework.event import listens_for
from ceam.framework.values import modifies_value, list_combiner, joint_value_post_processor, rescale_post_processor


class Disability:
    """Measures and assigns disability to the simulants."""
    def __init__(self):
        self.years_lived_with_disability = pd.Series([])

    def setup(self, builder):
        self.disability_weight = builder.value('disability_weight', list_combiner,
                                               lambda a: rescale_post_processor(joint_value_post_processor(a)))
        self.disability_weight.source = lambda index: [pd.Series(0.0, index=index)]

    @listens_for('initialize_simulants')
    def initialize_disability(self, event):
        self.years_lived_with_disability = self.years_lived_with_disability.append(
            pd.Series(0, index=event.index), verify_integrity=True)

    @listens_for('time_step__cleanup')
    def calculate_ylds(self, event):
        self.years_lived_with_disability[event.index] += self.disability_weight(event.index)

    @modifies_value('metrics')
    def metrics(self, index, metrics):
        metrics['years_lived_with_disability'] = self.years_lived_with_disability[index].sum()
        return metrics
