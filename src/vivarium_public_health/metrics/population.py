"""
===================
Population Observer
===================

This module contains tools for observing population counts during the
simulation.

"""
from collections import Counter

import pandas as pd

from .utilities import get_age_bins, get_population_counts


class PopulationObserver:
    """ An observer for population counts.

    By default, this counts the population at a particular sample date
    annually. It can be configured to bin the population into age groups and
    sexes by setting the ``by_age`` and ``by_sex`` flags, respectively. The
    sample date defaults to July 1st.  This is also configurable.

    The configuration defaults also contain a ``by_year`` flag, but it has
    no behavior associated with it.

    In the model specification, your configuration for this component should
    be specified as, e.g.:

    .. code-block:: yaml

        configuration:
            metrics:
                population:
                    by_age: True
                    by_sex: True
                    sample_date:
                        month: 4
                        day: 10

    """
    configuration_defaults = {
        'metrics': {
            'population': {
                'by_age': False,
                'by_year': False,
                'by_sex': False,
                'sample_date': {
                    'month': 7,
                    'day': 1,
                }
            }
        }
    }

    @property
    def name(self):
        return 'population_observer'

    def setup(self, builder):
        self.config = builder.configuration.metrics.population
        self.clock = builder.time.clock()
        self.age_bins = get_age_bins(builder)
        self.population = Counter()

        columns_required = ['tracked', 'alive']
        if self.config.by_age:
            columns_required += ['age']
        if self.config.by_sex:
            columns_required += ['sex']
        self.population_view = builder.population.get_view(columns_required)

        builder.event.register_listener('time_step__prepare', self.on_time_step_prepare)

        builder.value.register_value_modifier('metrics', self.metrics)

    def on_time_step_prepare(self, event):
        pop = self.population_view.get(event.index)

        if self.should_sample(event.time):
            population_counts = get_population_counts(pop, self.config.to_dict(), event.time, self.age_bins)
            self.population.update(population_counts)

    def should_sample(self, event_time: pd.Timestamp) -> bool:
        """Returns true if we should sample on this time step."""
        sample_date = pd.Timestamp(event_time.year, self.config.sample_date.month, self.config.sample_date.day)
        return self.clock() <= sample_date < event_time

    def metrics(self, index, metrics):
        metrics.update(self.population)
        return metrics

    def __repr__(self):
        return "PopulationObserver()"
