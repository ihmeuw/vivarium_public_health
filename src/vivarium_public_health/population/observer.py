from typing import List, Union, Tuple, Iterable
import pandas as pd
import numpy as np


def by_year(config):
    return ('observer' in config) and ('mortality' in config['observer']) and config['observer'].mortality.by_year


def to_years(time) -> float:
    return time / pd.Timedelta(days=365.25)


class MortalityObserver:
    configuration_defaults = {
        'by_year': False
    }

    def setup(self, builder):
        self.clock = builder.time.clock()
        self.step_size = builder.time.step_size()
        self.initial_pop_entrance_time = self.clock() - self.step_size()
        self.start_time = self.clock()
        self.configuration_defaults = {'observer': {'mortality': MortalityObserver.configuration_defaults['by_year']}}
        columns_required = ['tracked', 'alive', 'age', 'entrance_time', 'exit_time',
                            'cause_of_death', 'years_of_life_lost']
        self.age_bins = (builder.data.load('population.structure')[['age_group_start', 'age_group_end']]
                         .drop_duplicates()
                         .reset_index(drop=True))
        if builder.configuration.population.exit_age:
            self.age_bins = self.age_bins[self.age_bins.age_group_end <= builder.configuration.population.exit_age]

        self.population_view = builder.population.get_view(columns_required)
        self.by_year = by_year(builder.configuration)

        builder.value.register_value_modifier('metrics', self.metrics)

    def metrics(self, index, metrics):
        start_time = self.start_time
        end_time = self.clock()
        pop = self.population_view.get(index)
        years = range(start_time.year, end_time.year+1)
        pop.loc[pop.exit_time.isnull(), 'exit_time'] = end_time

        causes = pop.cause_of_death.unique().tolist()
        causes.remove('not_dead')
        causes.remove('death_due_to_other_causes')

        if not self.by_year:
            year = 'all_years'
            existing_at_start, born_in_sim = self.get_simulants_in_groups(self.initial_pop_entrance_time, end_time)
            total, born = self.count_deaths(year, existing_at_start, born_in_sim, self.age_bins, causes)
            total, born = self.count_person_time(year, existing_at_start, born_in_sim, self.age_bins, total, born)
        else:
            total, born = self._metrics_by_year(pop, years, start_time, end_time, causes)

        for ((label, age_group), count) in total.unstack().iteritems():
            metrics[f'age_group_{age_group}_{label}'] = count

        for ((label, age_group), count) in born.unstack().iteritems():
            metrics[f'age_group_{age_group}_{label}_among_born'] = count

        return metrics

    def _metrics_by_year(self, pop: pd.DataFrame, years: Iterable[int], start_time: pd.datetime,
                         end_time: pd.datetime, causes: List)-> Tuple:
        total_data = []
        born_data = []

        for year in years:
            first_entrance = self.initial_pop_entrance_time if year == start_time.year else pd.datetime(year, 1, 1)
            last_exit = end_time if year == end_time.year else pd.datetime(year, 12, 31)

            existing_at_start, born_in_sim = self.get_simulants_in_groups(first_entrance, last_exit, pop)
            total, born = self.count_deaths(year, existing_at_start, born_in_sim, self.age_bins, causes)
            total, born = self.count_person_time(year, existing_at_start, born_in_sim, self.age_bins, total, born)

            total_data.append(total)
            born_data.append(born)

        total = pd.concat(total_data)
        born = pd.concat(born_data)

        return total, born

    @staticmethod
    def get_simulants_in_groups(first_entrance: pd.datetime, last_exit: pd.datetime, pop: pd.DataFrame) -> Tuple:
        existing_at_start = pop.loc[(pop.entrance_time <= first_entrance) & (pop.exit_time > first_entrance)].copy()
        born_in_sim = pop.loc[(pop.entrance_time > first_entrance) & (pop.entrance_time < last_exit)].copy()

        years_till_exit = to_years(existing_at_start.exit_time - first_entrance)
        years_in_this_period = to_years(last_exit - first_entrance)

        existing_at_start['age_at_year_start'] = existing_at_start.age - years_till_exit
        existing_at_start['age_at_year_end'] = existing_at_start.age_at_year_start \
                                               + np.minimum(years_till_exit, years_in_this_period)

        max_newborn_years = to_years(last_exit - born_in_sim.entrance_time)
        born_in_sim['age_at_year_end'] = np.minimum(max_newborn_years, born_in_sim.age)

        return existing_at_start, born_in_sim

    @staticmethod
    def count_deaths(year: Union[int, str], existing_at_start: pd.DataFrame, born_in_sim: pd.DataFrame,
                     age_bins: pd.DataFrame, causes: List):

        causes = [f'death_due_to_{c}_{year}' for c in causes]
        causes.extend([f'death_due_to_other_causes_{year}'])
        causes.extend([f'total_deaths_{year}'])
        person_time = [f'person_time_{year}']

        frame_dict = dict()
        frame_dict.update({c: 0 for c in causes})
        frame_dict.update({p: 0 for p in person_time})

        total = pd.DataFrame(frame_dict, index=age_bins.index)
        born = pd.DataFrame(frame_dict, index=age_bins.index)

        for group, age_bin in age_bins.iterrows():
            start, end = age_bin.age_group_start, age_bin.age_group_end
            in_group = existing_at_start[(existing_at_start.age >= start) & (existing_at_start.age < end)]
            died = in_group[in_group.alive == 'dead']
            total.loc[group, f'total_deaths_{year}'] = len(died)
            cause_of_death = died.cause_of_death.value_counts()
            for cod, count in cause_of_death.iteritems():
                if 'death' not in cod:
                    cod = f'death_due_to_{cod}'
                cod += f'_{year}'
                total.loc[group, cod] += count

            in_group = born_in_sim[(born_in_sim.age >= start) & (born_in_sim.age < end)]
            died = in_group[in_group.alive == 'dead']
            born.loc[group, f'total_deaths_{year}'] = len(died)
            total.loc[group, f'total_deaths_{year}'] += len(died)
            cause_of_death = died.cause_of_death.value_counts()
            for cod, count in cause_of_death.iteritems():
                if 'death' not in cod:
                    cod = f'death_due_to_{cod}'
                cod += f'_{year}'
                born.loc[group, cod] += count
                total.loc[group, cod] += count

        return total, born

    @staticmethod
    def count_person_time(year: Union[int, str], existing_at_start: pd.DataFrame, born_in_sim: pd.DataFrame,
                          age_bins: pd.DataFrame, total: pd.DataFrame, born: pd.DataFrame):
        for group, age_bin in age_bins.iterrows():
            start, end = age_bin.age_group_start, age_bin.age_group_end

            alive_at_start_and_lived_in = existing_at_start[(existing_at_start.age_at_year_start < end)
                                                            & (existing_at_start.age_at_year_end >= start)]
            time_start = np.maximum(alive_at_start_and_lived_in.age_at_year_start, start)
            time_end = np.minimum(alive_at_start_and_lived_in.age_at_year_end, end)
            total.loc[group, f'person_time_{year}'] += (time_end - time_start).sum()

            born_and_lived_in = born_in_sim[(born_in_sim.age_at_year_end >= start) & (born_in_sim.age_at_year_end < end)]
            time_start = start
            time_end = np.minimum(born_and_lived_in.age_at_year_end, end)
            total.loc[group, f'person_time_{year}'] += (time_end - time_start).sum()
            born.loc[group, f'person_time_{year}'] += (time_end - time_start).sum()

        return total, born

