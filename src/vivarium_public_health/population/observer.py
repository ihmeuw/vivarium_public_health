from typing import List, Union, Tuple, Iterable
import pandas as pd
import numpy as np


def to_years(time) -> float:
    return time / pd.Timedelta(days=365.25)


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
        'mortality_observer': {
            'by_year': False
        }
    }

    def setup(self, builder):
        self.clock = builder.time.clock()
        self.step_size = builder.time.step_size()
        self.initial_pop_entrance_time = self.clock() - self.step_size()
        self.start_time = self.clock()
        columns_required = ['tracked', 'alive', 'age', 'entrance_time', 'exit_time',
                            'cause_of_death', 'years_of_life_lost']
        self.age_bins = builder.data.load('population.age_bins')
        exit_age = builder.configuration.population.exit_age
        if exit_age:
            self.age_bins = self.age_bins[self.age_bins.age_group_start < exit_age]
            self.age_bins.loc[self.age_bins.age_group_end > exit_age, 'age_group_end'] = exit_age

        self.population_view = builder.population.get_view(columns_required)
        self.by_year = builder.configuration.observer.mortality.by_year

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
            existing_at_start, born_in_sim = self.get_simulants_in_time_block(start_time, end_time, pop)
            total, born = self.count_deaths(year, existing_at_start, born_in_sim, self.age_bins, causes)
            total[f'person_time'], born[f'person_time'] = self.count_person_time(existing_at_start,
                                                                                 born_in_sim, self.age_bins)
        else:
            total, born = self._metrics_by_year(pop, years, start_time, end_time, causes)

        total = total.drop(['age_group_start', 'age_group_end'], axis=1).melt(id_vars=['year', 'age_group_name'])
        born = born.drop(['age_group_start', 'age_group_end'], axis=1).melt(id_vars=['year', 'age_group_name'])

        for _, row in total.iterrows():
            metrics[f'age_group_{row.age_group_name.replace(" ", "_")}_year_{row.year}_{row.variable}'] = row.value

        for _, row in born.iterrows():
            metrics[f'age_group_{row.age_group_name.replace(" ", "_")}_year_{row.year}_{row.variable}_among_born'] = row.value

        return metrics

    def _metrics_by_year(self, pop: pd.DataFrame, years: Iterable[int], start_time: pd.datetime,
                         end_time: pd.datetime, causes: List)-> Tuple:
        total_data = []
        born_data = []

        for year in years:
            first_entrance = self.start_time if year == start_time.year else pd.datetime(year, 1, 1)
            last_exit = end_time if year == end_time.year else pd.datetime(year, 12, 31)

            existing_before_block, born_in_block = self.get_simulants_in_time_block(first_entrance, last_exit, pop)
            total, born = self.count_deaths(year, existing_before_block, born_in_block, self.age_bins, causes)
            total[f'person_time'], born[f'person_time'] = self.count_person_time(existing_before_block, born_in_block,
                                                                                 self.age_bins)

            total_data.append(total)
            born_data.append(born)

        total = pd.concat(total_data)
        born = pd.concat(born_data)

        return total, born

    @staticmethod
    def get_simulants_in_time_block(block_start: pd.datetime, block_end: pd.datetime, pop: pd.DataFrame) -> Tuple:
        existing_before_block = pop.loc[(pop.entrance_time < block_start)].copy()
        born_in_block = pop.loc[(pop.entrance_time >= block_start) & (pop.entrance_time <= block_end)].copy()

        years_till_exit = to_years(existing_before_block.exit_time - block_start)
        years_in_this_period = to_years(block_end - block_start)

        # age is age at sim end when metrics are collected
        existing_before_block['age_at_year_start'] = existing_before_block.age - years_till_exit
        existing_before_block['age_at_year_end'] = existing_before_block.age_at_year_start \
                                               + np.minimum(years_till_exit, years_in_this_period)

        max_newborn_years = to_years(block_end - born_in_block.entrance_time)  # if simulant lived to/past end of block
        born_in_block['age_at_year_end'] = np.minimum(max_newborn_years, born_in_block.age)

        return existing_before_block, born_in_block

    @staticmethod
    def count_deaths(year: Union[int, str], existing_at_start: pd.DataFrame, born_in_sim: pd.DataFrame,
                     age_bins: pd.DataFrame, causes: List):

        causes = [f'death_due_to_{c}' for c in causes]
        causes.extend([f'death_due_to_other_causes'])
        causes.extend([f'total_deaths'])

        frame_dict = dict()
        frame_dict.update({c: 0 for c in causes})

        total = pd.DataFrame(frame_dict, index=age_bins.index).join(age_bins)
        born = pd.DataFrame(frame_dict, index=age_bins.index).join(age_bins)
        total['year'] = year
        born['year'] = year

        for group, age_bin in age_bins.iterrows():
            start, end = age_bin.age_group_start, age_bin.age_group_end
            # because only counting dead simulants, age is frozen at time of death
            in_group = existing_at_start[(existing_at_start.age >= start) & (existing_at_start.age < end)]
            died = in_group[in_group.alive == 'dead']
            total.loc[group, f'total_deaths'] = len(died)
            cause_of_death = died.cause_of_death.value_counts()
            for cod, count in cause_of_death.iteritems():
                if 'death' not in cod:
                    cod = f'death_due_to_{cod}'
                total.loc[group, cod] = count

            in_group = born_in_sim[(born_in_sim.age >= start) & (born_in_sim.age < end)]
            died = in_group[in_group.alive == 'dead']
            born.loc[group, f'total_deaths'] = len(died)
            total.loc[group, f'total_deaths'] += len(died)
            cause_of_death = died.cause_of_death.value_counts()
            for cod, count in cause_of_death.iteritems():
                if 'death' not in cod:
                    cod = f'death_due_to_{cod}'
                born.loc[group, cod] = count
                total.loc[group, cod] += count

        return total, born

    @staticmethod
    def count_person_time(existing_before_block: pd.DataFrame, born_in_block: pd.DataFrame, age_bins: pd.DataFrame):

        total, born = pd.Series(0, index=age_bins.index), pd.Series(0, index=age_bins.index)

        for group, age_bin in age_bins.iterrows():
            start, end = age_bin.age_group_start, age_bin.age_group_end

            # ALIVE BEFORE BLOCK
            in_age_group = existing_before_block[(existing_before_block.age_at_year_start < end)
                                                 & (existing_before_block.age_at_year_end >= start)]

            age_start = np.maximum(in_age_group.age_at_year_start, start)
            age_end = np.minimum(in_age_group.age_at_year_end, end)
            total.loc[group] += (age_end - age_start).sum()

            # BORN IN BLOCK
            in_age_group = born_in_block[(born_in_block.age_at_year_end >= start)]
            age_start = np.maximum(0, start)
            age_end = np.minimum(in_age_group.age_at_year_end, end)
            total.loc[group] += (age_end - age_start).sum()
            born.loc[group] += (age_end - age_start).sum()

        return total, born
