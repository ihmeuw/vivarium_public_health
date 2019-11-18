"""
===================
Disability Observer
===================

This module contains tools for observing years lived with disability (YLDs)
in the simulation.

"""
from collections import Counter

import pandas as pd
from vivarium.framework.values import list_combiner, union_post_processor, rescale_post_processor

from vivarium_public_health.disease import DiseaseState, RiskAttributableDisease
from .utilities import get_age_bins, get_years_lived_with_disability


class DisabilityObserver:
    """Counts years lived with disability.

    By default, this counts both aggregate and cause-specific years lived
    with disability over the full course of the simulation. It can be
    configured to bin the cause-specific YLDs into age groups, sexes, and years
    by setting the ``by_age``, ``by_sex``, and ``by_year`` flags, respectively.

    In the model specification, your configuration for this component should
    be specified as, e.g.:

    .. code-block:: yaml

        configuration:
            metrics:
                disability:
                    by_age: True
                    by_year: False
                    by_sex: True

    """
    configuration_defaults = {
        'metrics': {
            'disability': {
                'by_age': False,
                'by_year': False,
                'by_sex': False,
            }
        }
    }

    @property
    def name(self):
        return 'disability_observer'

    def setup(self, builder):
        self.config = builder.configuration.metrics.disability
        self.age_bins = get_age_bins(builder)
        self.clock = builder.time.clock()
        self.step_size = builder.time.step_size()
        self.causes = [c.state_id
                       for c in builder.components.get_components_by_type((DiseaseState, RiskAttributableDisease))]
        self.years_lived_with_disability = Counter()
        self.disability_weight_pipelines = {cause: builder.value.get_value(f'{cause}.disability_weight')
                                            for cause in self.causes}

        self.disability_weight = builder.value.register_value_producer(
            'disability_weight',
            source=lambda index: [pd.Series(0.0, index=index)],
            preferred_combiner=list_combiner,
            preferred_post_processor=_disability_post_processor)

        columns_required = ['tracked', 'alive', 'years_lived_with_disability']
        if self.config.by_age:
            columns_required += ['age']
        if self.config.by_sex:
            columns_required += ['sex']
        self.population_view = builder.population.get_view(columns_required)
        builder.population.initializes_simulants(self.initialize_disability,
                                                 creates_columns=['years_lived_with_disability'])
        # FIXME: The state table is modified before the clock advances.
        # In order to get an accurate representation of person time we need to look at
        # the state table before anything happens.
        builder.event.register_listener('time_step__prepare', self.on_time_step_prepare)
        builder.value.register_value_modifier('metrics', modifier=self.metrics)

    def initialize_disability(self, pop_data):
        self.population_view.update(pd.Series(0., index=pop_data.index, name='years_lived_with_disability'))

    def on_time_step_prepare(self, event):
        pop = self.population_view.get(event.index, query='tracked == True and alive == "alive"')
        ylds_this_step = get_years_lived_with_disability(pop, self.config.to_dict(),
                                                         self.clock().year, self.step_size(),
                                                         self.age_bins, self.disability_weight_pipelines, self.causes)
        self.years_lived_with_disability.update(ylds_this_step)

        pop.loc[:, 'years_lived_with_disability'] += self.disability_weight(pop.index)
        self.population_view.update(pop)

    def metrics(self, index, metrics):
        total_ylds = self.population_view.get(index)['years_lived_with_disability'].sum()
        metrics['years_lived_with_disability'] = total_ylds
        metrics.update(self.years_lived_with_disability)
        return metrics

    def __repr__(self):
        return "DisabilityObserver()"


def _disability_post_processor(value, step_size):
    return rescale_post_processor(union_post_processor(value, step_size), step_size)
