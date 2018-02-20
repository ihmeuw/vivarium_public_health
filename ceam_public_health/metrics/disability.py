import pandas as pd

from vivarium.framework.event import listens_for
from vivarium.framework.values import list_combiner, joint_value_post_processor, rescale_post_processor


def _disability_post_processor(value, step_size):
    return rescale_post_processor(joint_value_post_processor(value, step_size), step_size)


class Disability:
    """Measures and assigns disability to the simulants."""
    def __init__(self):
        self.years_lived_with_disability = pd.Series([])

    def setup(self, builder):
        self.disability_weight = builder.value.register_value_producer(
            'disability_weight',
            source=lambda index: [pd.Series(0.0, index=index)],
            preferred_combiner=list_combiner,
            preferred_post_processor=_disability_post_processor)
        builder.value.register_value_modifier('metrics', modifier=self.metrics)

    @listens_for('initialize_simulants')
    def initialize_disability(self, event):
        self.years_lived_with_disability = self.years_lived_with_disability.append(
            pd.Series(0, index=event.index))

    @listens_for('collect_metrics')
    def calculate_ylds(self, event):
        self.years_lived_with_disability[event.index] += self.disability_weight(event.index)

    def metrics(self, index, metrics):
        metrics['years_lived_with_disability'] = self.years_lived_with_disability[index].sum()
        return metrics
