import typing

import numpy as np

if typing.TYPE_CHECKING:
    from vivarium.framework.engine import Builder


class Metrics:

    configuration_defaults = {
        'metrics': {
            'stratification_params': []
        }
    }

    @property
    def name(self):
        # Fixme: Avoiding name collision for now.
        return 'metrics2'

    def setup(self, builder: 'Builder'):
        self.config = builder.configuration.metrics
        if 'age_group' in self.config.stratification_params:
            age_bins = self.get_age_bins(builder)
            target_col = 'age'
            result_col = 'age_group'
            bins = age_bins.age_start.to_list() + [age_bins.age_end.iloc[-1]]
            labels = age_bins.age_group_name.str.replace(' ', '_').str.lower().to_list()
            builder.results.add_binner(target_col, result_col, bins, labels, include_lowest=True)
        if 'year' in self.config.stratification_params:
            for time_type in ['current', 'event', 'entrance', 'exit']:
                builder.results.add_mapper(f'{time_type}_time', f'{time_type}_year',
                                           lambda x: x.dt.year, is_vectorized=True)

    def get_age_bins(self, builder: 'Builder'):
        age_bins = builder.data.load('population.age_bins')

        # Works based on the fact that currently only models with age_start = 0 can include fertility
        age_start = builder.configuration.population.age_start
        min_bin_start = age_bins.age_start[np.asscalar(np.digitize(age_start, age_bins.age_end))]
        age_bins = age_bins[age_bins.age_start >= min_bin_start]
        age_bins.loc[age_bins.age_start < age_start, 'age_start'] = age_start

        exit_age = builder.configuration.population.exit_age
        if exit_age:
            age_bins = age_bins[age_bins.age_start < exit_age]
            age_bins.loc[age_bins.age_end > exit_age, 'age_end'] = exit_age
        return age_bins




