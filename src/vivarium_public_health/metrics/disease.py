"""
================
Disease Observer
================

This module contains tools for observing disease incidence and prevalence
in the simulation.

"""
from collections import Counter

import pandas as pd

from .utilities import get_age_bins, get_disease_event_counts, get_susceptible_person_time, get_prevalent_cases


class DiseaseObserver:
    """Observes disease counts, person time, and prevalent cases for a cause.

    By default, this observer computes aggregate susceptible person time
    and counts of disease cases over the entire simulation.  It can be
    configured to bin these into age_groups, sexes, and years by setting
    the ``by_age``, ``by_sex``, and ``by_year`` flags, respectively.

    It also can record prevalent cases on a particular sample date each year,
    though by default this is disabled. These will also be binned based on the
    flags set for the observer. Additionally, the sample date is configurable
    and defaults to July 1st of each year.

    In the model specification, your configuration for this component should
    be specified as, e.g.:

    .. code-block:: yaml

        configuration:
            metrics:
                {YOUR_DISEASE_NAME}_observer:
                    by_age: True
                    by_year: False
                    by_sex: True
                    sample_prevalence:
                        sample: True
                        date:
                            month: 4
                            day: 10

    """
    configuration_defaults = {
        'metrics': {
            'disease_observer': {
                'by_age': False,
                'by_year': False,
                'by_sex': False,
                'sample_prevalence': {
                    'sample': False,
                    'date': {
                        'month': 7,
                        'day': 1,
                    }
                },
            }
        }
    }

    def __init__(self, disease: str):
        self.disease = disease
        self.configuration_defaults = {
            'metrics': {f'{disease}_observer': DiseaseObserver.configuration_defaults['metrics']['disease_observer']}
        }

    @property
    def name(self):
        return f'disease_observer.{self.disease}'

    def setup(self, builder):
        self.config = builder.configuration['metrics'][f'{self.disease}_observer']
        self.clock = builder.time.clock()
        self.age_bins = get_age_bins(builder)
        self.counts = Counter()
        self.person_time = Counter()
        self.prevalence = Counter()

        columns_required = ['alive', f'{self.disease}', f'{self.disease}_event_time']
        if self.config.by_age:
            columns_required += ['age']
        if self.config.by_sex:
            columns_required += ['sex']
        self.population_view = builder.population.get_view(columns_required)

        builder.value.register_value_modifier('metrics', self.metrics)
        # FIXME: The state table is modified before the clock advances.
        # In order to get an accurate representation of person time we need to look at
        # the state table before anything happens.
        builder.event.register_listener('time_step__prepare', self.on_time_step_prepare)
        builder.event.register_listener('collect_metrics', self.on_collect_metrics)

    def on_time_step_prepare(self, event):
        pop = self.population_view.get(event.index)
        # Ignoring the edge case where the step spans a new year.
        # Accrue all counts and time to the current year.
        person_time_this_step = get_susceptible_person_time(pop, self.config.to_dict(), self.disease,
                                                            self.clock().year, event.step_size, self.age_bins)
        self.person_time.update(person_time_this_step)

        if self._should_sample(event.time):
            point_prevalence = get_prevalent_cases(pop, self.config.to_dict(), self.disease, event.time, self.age_bins)
            self.prevalence.update(point_prevalence)

    def on_collect_metrics(self, event):
        pop = self.population_view.get(event.index)
        disease_events_this_step = get_disease_event_counts(pop, self.config.to_dict(), self.disease,
                                                            event.time, self.age_bins)
        self.counts.update(disease_events_this_step)

    def _should_sample(self, event_time: pd.Timestamp) -> bool:
        """Returns true if we should sample on this time step."""
        should_sample = self.config.sample_prevalence.sample
        if should_sample:
            sample_date = pd.Timestamp(year=event_time.year, **self.config.sample_prevalence.date.to_dict())
            should_sample &= self.clock() <= sample_date < event_time
        return should_sample

    def metrics(self, index, metrics):
        metrics.update(self.counts)
        metrics.update(self.person_time)
        metrics.update(self.prevalence)
        return metrics

    def __repr__(self):
        return "DiseaseObserver()"
