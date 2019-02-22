from collections import defaultdict

import pandas as pd

from .utilities import get_age_bins, get_output_template, to_years


class DiseaseObserver:
    """Observes disease counts and person time for a single cause.

    By default, this observer computes aggregate susceptible person time
    and counts of disease cases over the entire simulation.  It can be
    configured to bin these into age_groups, sexes, and years by setting
    the ``by_age``, ``by_sex``, and ``by_year`` flags, respectively.

    """
    configuration_defaults = {
        'disease_observer': {
            'by_age': False,
            'by_year': False,
            'by_sex': False,
        }
    }

    def __init__(self, disease: str):
        self.disease = disease
        self.name = f'{self.disease}_observer'
        self.configuration_defaults = {self.name: DiseaseObserver.configuration_defaults['disease_observer']}

    def setup(self, builder):
        self.config = builder.configuration[self.name]

        self.clock = builder.time.clock()

        self.output_template = get_output_template(**self.config.to_dict())

        self.age_bins = get_age_bins(builder)
        self.counts = defaultdict(int)
        self.person_time = defaultdict(float)

        columns_required = ['alive', 'exit_time', f'{self.disease}', f'{self.disease}_event_time']
        if self.config.by_age:
            columns_required += ['age']
        if self.config.by_sex:
            columns_required += ['sex']

        self.population_view = builder.population.get_view(columns_required, query='alive == "alive"')

        builder.value.register_value_modifier('metrics', self.metrics)
        # FIXME: The state table is modified before the clock advances.
        # In order to get an accurate representation of person time and disease
        # counts, we need to look at the state table before anything happens.
        builder.event.register_listener('time_step__prepare', self.on_time_step_prepare)

    def on_time_step_prepare(self, event):
        pop = self.population_view.get(event.index)

        # Ignoring the edge case where the step spans a new year.
        # Accrue all counts and time to the current year.
        key = self.output_template.safe_substitute(year=self.clock().year)
        count_key = key.safe_substitute(measure=f'{self.disease}_counts')
        person_time_key = key.safe_substitute(measure=f'{self.disease}_susceptible_person_time')

        filter_string = f'{self.disease} == susceptible_to_{self.disease}'

        if self.config.by_age:
            ages = self.age_bins.iterrows()
            filter_string += ' and ({age_group_start} <= age) and (age < {age_group_end})'
        else:
            ages = [('all_ages', pd.Series({'age_group_start': None, 'age_group_end': None}))]

        if self.config.by_sex:
            sexes = ['Male', 'Female']
            filter_string += ' and sex == {sex}'
        else:
            sexes = ['Both']

        for group, age_group in ages:
            start, end = age_group.age_group_start, age_group.age_group_end
            for sex in sexes:
                filter_kwargs = {'age_group_start': start, 'age_group_end': end, 'sex': sex}
                group_count_key = count_key.safe_substitute(**filter_kwargs)
                group_person_time_key = person_time_key.safe_substitute(**filter_kwargs)
                group_filter = filter_string.format(**filter_kwargs)

                in_group = pop.query(group_filter)

                self.counts[group_count_key] += len(in_group)
                self.person_time[group_person_time_key] += len(in_group) * to_years(event.step_size)

    def metrics(self, index, metrics):
        metrics.update(self.counts)
        metrics.update(self.person_time)
        return metrics
