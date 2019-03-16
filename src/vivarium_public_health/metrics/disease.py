from collections import Counter

from .utilities import get_age_bins, get_disease_event_counts, get_susceptible_person_time


class DiseaseObserver:
    """Observes disease counts and person time for a single cause.

    By default, this observer computes aggregate susceptible person time
    and counts of disease cases over the entire simulation.  It can be
    configured to bin these into age_groups, sexes, and years by setting
    the ``by_age``, ``by_sex``, and ``by_year`` flags, respectively.

    In the model specification, your configuration for this component should
    be specified as, e.g.:

    configuration:
        metrics:
            YOUR_DISEASE_NAME:
                by_age: True
                by_year: False
                by_sex: True

    """
    configuration_defaults = {
        'metrics': {
            'disease_observer': {
                'by_age': False,
                'by_year': False,
                'by_sex': False,
            }
        }
    }

    def __init__(self, disease: str):
        self.disease = disease
        self.name = f'{self.disease}_observer'
        self.configuration_defaults = {
            'metrics': {self.name: DiseaseObserver.configuration_defaults['metrics']['disease_observer']}
        }

    def setup(self, builder):
        self.config = builder.configuration['metrics'][self.name]
        self.clock = builder.time.clock()
        self.age_bins = get_age_bins(builder)
        self.counts = Counter()
        self.person_time = Counter()

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

    def on_collect_metrics(self, event):
        pop = self.population_view.get(event.index)
        disease_events_this_step = get_disease_event_counts(pop, self.config.to_dict(), self.disease,
                                                            event.time, self.age_bins)
        self.counts.update(disease_events_this_step)

    def metrics(self, index, metrics):
        metrics.update(self.counts)
        metrics.update(self.person_time)
        return metrics
