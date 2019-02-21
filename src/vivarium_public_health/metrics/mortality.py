import pandas as pd
import numpy as np

from .utilities import get_age_bins, clean_cause_of_death, to_years


class MortalityObserver:
    """ An observer for total and cause specific deaths during simulation.
    This component counts total and cause specific deaths in the population
    as well as person time (the time spent alive and tracked in the
    simulation).
    The data is discretized by age groups and optionally by year.
    """
    configuration_defaults = {
        'metrics': {
            'mortality': {
                # TODO: Implement by_sex and by_age flags
                'by_year': False
            }
        }
    }

    def setup(self, builder):
        self.by_year = builder.configuration.metrics.mortality.by_year
        self.clock = builder.time.clock()
        self.step_size = builder.time.step_size()
        self.initial_pop_entrance_time = self.clock() - self.step_size()
        self.start_time = self.clock()
        self.age_bins = get_age_bins(builder)

        columns_required = ['tracked', 'alive', 'age', 'entrance_time', 'exit_time',
                            'cause_of_death', 'years_of_life_lost']
        self.population_view = builder.population.get_view(columns_required)

        builder.value.register_value_modifier('metrics', self.metrics)

    def metrics(self, index, metrics):
        pop = self.population_view.get(index)
        pop.loc[pop.exit_time.isnull(), 'exit_time'] = self.clock()

        pop = clean_cause_of_death(pop)
        born_in_sim = pop[pop.entrance_time > self.start_time]
        metrics.update(self.get_metrics(pop))
        metrics.update(self.get_metrics(born_in_sim, among_born=True))

        if self.by_year:
            for year in range(self.start_time.year, self.clock().year + 1):
                metrics.update(self.get_metrics(pop, year))
                metrics.update(self.get_metrics(born_in_sim, year, among_born=True))

        return metrics

    def get_metrics(self, pop, year=None, among_born=False):
        if year is not None:
            start, end = pd.Timestamp(f'1-1-{year}'), pd.Timestamp(f'1-1-{year + 1}')
        else:
            start, end = pd.Timestamp(f'1-1-1900'), pd.Timestamp(f'1-1-2100')

        out = {}

        data = count_deaths(pop, self.age_bins, start, end)
        data['person_time'] = count_person_time(pop, self.age_bins, start, end)
        out.update(self.format_output(data, year, among_born))

        return out

    def format_output(self, data, year, among_born):
        out = {}
        born_flag = 'born_in_sim' if among_born else 'all_simulants'
        year_flag = year if year is not None else 'all_years'
        for i, row in data.iterrows():
            age_group_name = self.age_bins.at[i, 'age_group_name'].replace(' ', '_')
            for variable, value in row.iteritems():
                label = f'{variable}_in_{year_flag}_among_{age_group_name}_{born_flag}'
                out[label] = value
        return out


def count_person_time(pop, age_bins, start_time, end_time):
    lived_in_span = pop[(start_time <= pop.exit_time) & (pop.entrance_time < end_time)]
    # The right way to do this is np.maximum/np.minimum,
    # but there's some bug in pandas that causes that to break.
    entrance_time = lived_in_span.entrance_time
    exit_time = lived_in_span.exit_time
    exit_time.loc[end_time < exit_time] = end_time

    years_in_span = to_years(exit_time - entrance_time)
    lived_in_span['age_at_start'] = np.maximum(lived_in_span.age - years_in_span, 0)

    data = pd.Series(0, index=age_bins.index)

    for group, age_bin in age_bins.iterrows():
        start, end = age_bin.age_group_start, age_bin.age_group_end
        in_group = lived_in_span[(start < lived_in_span.age) & (lived_in_span.age_at_start < end)]
        age_start = np.maximum(in_group.age_at_start, start)
        age_end = np.minimum(in_group.age, end)
        data.loc[group] += (age_end - age_start).sum()

    return data


def count_deaths(pop, age_bins, start_time, end_time):
    deaths = pop[(pop.alive == 'dead') & (start_time <= pop.exit_time) & (pop.exit_time < end_time)]

    causes = [c for c in pop.cause_of_death.unique() if c != 'not_dead'] + ['total_deaths']
    data = pd.DataFrame({c: 0 for c in causes}, index=age_bins.index)

    for group, age_bin in age_bins.iterrows():
        start, end = age_bin.age_group_start, age_bin.age_group_end
        in_group = deaths[(start <= deaths.age) & (deaths.age < end)]

        data.loc[group, f'total_deaths'] = len(in_group)
        cause_of_death = in_group.cause_of_death.value_counts()
        for cod, count in cause_of_death.iteritems():
            data.loc[group, cod] = count

    return data
