import pandas as pd

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

        self.population_view = builder.population.get_view(['years_lived_with_disability'])
        builder.population.initializes_simulants(self.initialize_disability,
                                                 creates_columns=['years_lived_with_disability'])
        builder.event.register_listener('collect_metrics', self.calculate_ylds)

    def initialize_disability(self, pop_data):
        self.population_view.update(pd.Series(0., index=pop_data.index))

    def calculate_ylds(self, event):
        disability = self.population_view.get(event.index)['years_lived_with_disability']
        disability += self.disability_weight(event.index)
        self.population_view.update(disability)

    def metrics(self, index, metrics):
        metrics['years_lived_with_disability'] = self.population_view.get(index)['years_lived_with_disability'].sum()
        return metrics
