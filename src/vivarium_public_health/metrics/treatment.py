from collections import defaultdict

from .utilities import get_age_bins, get_output_template, get_group_counts


class TreatmentObserver:
    """Observes dose counts for a single treatment.

    This component is intended for use with the ``MassTreatmentCampaign``
    component in ``vivarium_public_health``.  It expects the names of the
    doses to be supplied under the configuration key
    ``{treatment_name}.doses`` and should work with any component that
    specifies its doses in that manner.

    By default, this observer computes aggregate dose counts over the entire
    simulation.  It can be configured to bin these into age_groups, sexes,
    and years by setting the ``by_age``, ``by_sex``, and ``by_year``
    flags, respectively.

    In the model specification, your configuration for this component should
    be specified as, e.g.:

    configuration:
        metrics:
            YOUR_TREATMENT_NAME:
                by_age: True
                by_year: False
                by_sex: True

    """

    configuration_defaults = {
        'metrics': {
            'treatment_observer': {
                'by_age': False,
                'by_year': False,
                'by_sex': False,
            }
        }
    }

    def __init__(self, treatment: str):
        self.treatment = treatment
        self.name = f'{self.treatment}_observer'
        self.configuration_defaults = {
            'metrics': {self.name: TreatmentObserver.configuration_defaults['metrics']['treatment_observer']}
        }

    def setup(self, builder):
        self.config = builder.configuration['metrics'][self.name]
        self.doses = builder.configuration[self.treatment]['doses']

        self.output_template = get_output_template(**self.config.to_dict())

        self.age_bins = get_age_bins(builder)
        self.counts = defaultdict(int)

        columns_required = ['alive', f'{self.treatment}_current_dose_event_time', f'{self.treatment}_current_dose']
        if self.config.by_age:
            columns_required += ['age']
        if self.config.by_sex:
            columns_required += ['sex']

        self.population_view = builder.population.get_view(columns_required)

        builder.value.register_value_modifier('metrics', self.metrics)
        builder.event.register_listener('on_collect_metrics', self.on_collect_metrics)

    def on_collect_metrics(self, event):
        pop = self.population_view.get(event.index)

        base_key = self.output_template.safe_substitute(year=event.time.year)

        for dose in self.doses:
            base_filter = (f'{self.treatment}_current_dose == {dose} and '
                           f'{self.treatment}_current_dose_event_time == {event.time}')
            group_counts = get_group_counts(pop, base_filter, base_key, self.config.to_dict(), self.age_bins)

            for key, count in group_counts.items():
                key = base_key.safe_substitute(measure=f'{self.treatment}_{dose}_count')
                self.counts[key] += count

    def metrics(self, index, metrics):
        metrics.update(self.counts)
        return metrics
