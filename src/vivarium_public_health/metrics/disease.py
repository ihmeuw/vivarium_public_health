from collections import defaultdict

from .utilities import get_age_bins, get_output_template, to_years, get_group_counts


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

        self.output_template = get_output_template(**self.config.to_dict())

        self.age_bins = get_age_bins(builder)
        self.counts = defaultdict(int)
        self.person_time = defaultdict(float)

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
        builder.event.register_listener('on_collect_metrics', self.on_collect_metrics)

    def on_time_step_prepare(self, event):
        pop = self.population_view.get(event.index, query='alive == "alive"')

        # Ignoring the edge case where the step spans a new year.
        # Accrue all counts and time to the current year.
        base_key = self.output_template.safe_substitute(year=self.clock().year)
        base_filter = f'{self.disease} == susceptible_to_{self.disease}'

        group_counts = get_group_counts(pop, base_filter, base_key, self.config.to_dict(), self.age_bins)

        for key, count in group_counts.items():
            person_time_key = key.safe_substitute(measure=f'{self.disease}_susceptible_person_time')
            self.person_time[person_time_key] += count * to_years(event.step_size)

    def on_collect_metrics(self, event):
        pop = self.population_view.get(event.index)

        base_key = self.output_template.safe_substitute(year=event.time.year)
        base_filter = f'{self.disease}_event_time == {event.time}'

        group_counts = get_group_counts(pop, base_filter, base_key, self.config.to_dict(), self.age_bins)

        for key, count in group_counts.items():
            count_key = key.safe_substitute(measure=f'{self.disease}_counts')
            self.counts[count_key] += count

    def metrics(self, index, metrics):
        metrics.update(self.counts)
        metrics.update(self.person_time)
        return metrics
