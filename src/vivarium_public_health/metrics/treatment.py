"""
==================
Treatment Observer
==================

This module contains tools for observing multi-dose treatment counts
in the simulation.

"""
from collections import Counter

from .utilities import get_age_bins, get_treatment_counts


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

    .. code-block:: yaml

        configuration:
            metrics:
                {YOUR_TREATMENT_NAME}_observer:
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
        self.configuration_defaults = {
            'metrics': {self.name: TreatmentObserver.configuration_defaults['metrics']['treatment_observer']}
        }

    @property
    def name(self):
        return f'treatment_observer.{self.treatment}'

    def setup(self, builder):
        self.config = builder.configuration['metrics'][self.name]
        self.doses = builder.configuration[self.treatment]['doses']
        self.age_bins = get_age_bins(builder)
        self.counts = Counter()

        columns_required = ['alive', f'{self.treatment}_current_dose_event_time', f'{self.treatment}_current_dose']
        if self.config.by_age:
            columns_required += ['age']
        if self.config.by_sex:
            columns_required += ['sex']
        self.population_view = builder.population.get_view(columns_required)

        builder.value.register_value_modifier('metrics', self.metrics)
        builder.event.register_listener('collect_metrics', self.on_collect_metrics)

    def on_collect_metrics(self, event):
        pop = self.population_view.get(event.index)
        dose_counts_this_step = get_treatment_counts(pop, self.config.to_dict(), self.treatment, self.doses, event.time, self.age_bins)
        self.counts.update(dose_counts_this_step)

    def metrics(self, index, metrics):
        metrics.update(self.counts)
        return metrics

    def __repr__(self):
        return f"TreatmentObserver({self.treatment})"
