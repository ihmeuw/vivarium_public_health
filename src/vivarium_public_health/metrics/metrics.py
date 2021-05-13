import typing

import numpy as np
import pandas as pd

if typing.TYPE_CHECKING:
    from vivarium.framework.engine import Builder


class Metrics:

    configuration_defaults = {
        'metrics': {
            # Make a global list of stratification parameters.
            'stratification_params': []
        }
    }

    @property
    def name(self):
        # TODO: Avoiding naming collision with default simulation component
        # currently injected by the framework.
        return 'metrics2'

    def setup(self, builder: 'Builder'):
        self.config = builder.configuration.metrics
        self.age_bins = self.get_age_bins(builder)
        self.clock = builder.time.clock()
        self.step_size = builder.time.step_size()

        required_columns = ['tracked', 'age', 'entrance_time', 'exit_time']
        self.population_view = builder.population.get_view(required_columns)
        builder.results.add_mapping_strategy('age_group', self.make_age_group)
        builder.results.add_mapping_strategy('current_year', self.make_current_year)
        builder.results.add_mapping_strategy('event_year', self.make_event_year)
        builder.results.add_mapping_strategy('entrance_year', self.make_entrance_year)
        builder.results.add_mapping_strategy('exit_year', self.make_exit_year)

        builder.results.add_default_grouping_columns(self.config.stratification_params)

    def make_age_group(self, index: pd.Index) -> pd.Series:
        pop = self.population_view.get(index)
        bins = self.age_bins.age_start.to_list() + [self.age_bins.age_end.iloc[-1]]
        labels = self.age_bins.age_group_name.str.replace(' ', '_').str.lower().to_list()
        age_group = pd.cut(pop.age, bins, labels).rename('age_group')
        return age_group

    def make_current_year(self, index: pd.Index) -> pd.Series:
        return pd.Series(self.clock().year, index=index, name='current_year')

    def make_event_year(self, index: pd.Index) -> pd.Series:
        return pd.Series((self.clock() + self.step_size()).year, index=index, name='event_year')

    def make_entrance_year(self, index: pd.Index) -> pd.Series:
        pop = self.population_view.get(index)
        return pop.entrance_time.dt.year.rename('entrance_year')

    def make_exit_year(self, index: pd.Index) -> pd.Series:
        pop = self.population_view.get(index)
        return pop.exit_time.dt.year.rename('exit_year')

    @staticmethod
    def get_age_bins(builder: 'Builder'):
        # Pulled directly from the normal metrics utilities. should only be needed here in the future.
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
