import pandas as pd
import numpy as np

from .utilities import get_age_bins, get_sexes


class MortalityObserver:
    """ An observer for total and cause specific deaths during simulation.
    This component by default observes the total number of deaths out of
    total population and total person time that each simulant spent
    during the simulation until it exits.

    These data are categorized by age groups and causes and aggregated over
    total population as well as the population who were born during
    the simulation.

    By default, we also aggregate over time. If by_year flag is turned on,
    we aggregate the data by each year.

    """
    configuration_defaults = {
        'metrics': {
            'mortality': {
                'by_year': False
            }
        }
    }

    def setup(self, builder):
        self.name = 'mortality_observer'

        self.by_year = builder.configuration.metrics.mortality.by_year
        self.clock = builder.time.clock()
        self.step_size = builder.time.step_size()
        self.initial_pop_entrance_time = self.clock() - self.step_size()
        self.start_time = self.clock()
        self.age_bins = get_age_bins(builder)
        self.sexes = get_sexes()

        columns_required = ['tracked', 'alive', 'age', 'sex', 'entrance_time',
                            'exit_time', 'cause_of_death', 'years_of_life_lost']
        self.population_view = builder.population.get_view(columns_required)

        builder.value.register_value_modifier('metrics', self.metrics)

    def metrics(self, index, metrics):
        pop = self.population_view.get(index)
        pop.loc[pop.exit_time.isnull(), 'exit_time'] = self.clock()

        pop = clean_cause_of_death(pop)
        born_in_sim = pop[pop.entrance_time > self.start_time]
        metrics.update(self.get_metrics(pop))
        metrics.update(self.get_metrics(born_in_sim, among_born=True))

        for sex in self.sexes:
            if self.by_year:
                for year in range(self.start_time.year, self.clock().year + 1):
                    metrics.update(self.get_metrics(pop, year=year, sex=sex))
                    metrics.update(self.get_metrics(born_in_sim, year=year, sex=sex, among_born=True))
            else:
                metrics.update(self.get_metrics(pop, sex=sex))
                metrics.update(self.get_metrics(born_in_sim, sex=sex, among_born=True))

        return metrics

    def get_metrics(self, pop, year=None, sex=None, among_born=False):
        if year is not None:
            start, end = pd.Timestamp(f'1-1-{year}'), pd.Timestamp(f'1-1-{year + 1}')
        else:
            start, end = pd.Timestamp(f'1-1-1900'), pd.Timestamp(f'1-1-2100')

        if sex is not None:
            pop = pop.loc[pop.sex == sex]

        out = {}
        data = count_deaths(pop, self.age_bins, start, end)
        data['person_time'] = count_person_time(pop, self.age_bins, start, end)
        out.update(self.format_output(data, year, sex, among_born))

        return out

    def format_output(self, data, year, sex, among_born):
        out = {}
        born_flag = 'born_in_sim' if among_born else 'all_simulants'
        year_flag = year if year is not None else 'all_years'
        sex_flag = sex if sex is not None else 'both_sexes'
        for i, row in data.iterrows():
            age_group_name = self.age_bins.at[i, 'age_group_name'].replace(' ', '_')
            for variable, value in row.iteritems():
                label = f'{variable}_for_{sex_flag}_in_{year_flag}_among_{age_group_name}_{born_flag}'
                out[label] = value
        return out


def count_person_time(pop, age_bins, start_time, end_time):
    lived_in_span = pop[(start_time <= pop.exit_time) & (pop.entrance_time < end_time)]
    # The right way to do this is np.maximum/np.minimum,
    # but there's some bug in pandas that causes that to break.
    entrance_time = lived_in_span.entrance_time
    exit_time = lived_in_span.exit_time
    exit_time.loc[end_time < exit_time] = end_time

    years_in_span = (exit_time - entrance_time) / pd.Timedelta(days=365.25)
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


def clean_cause_of_death(pop):

    def _clean(cod):
        if 'death' in cod or 'dead' in cod:
            pass
        else:
            cod = f'death_due_to_{cod}'
        return cod

    pop.cause_of_death = pop.cause_of_death.apply(_clean)
    return pop


def to_years(time) -> float:
    return time / pd.Timedelta(days=365.25)
