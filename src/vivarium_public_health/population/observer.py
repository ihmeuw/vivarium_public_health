from typing import List, Union
import pandas as pd
import numpy as np


def by_year(config):
    return ('observer' in config) and ('mortality' in config['observer']) and config['observer'].mortality.by_year


def to_years(time: pd.datetime) -> float:
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
        exit_age = builder.configuration.exit_age
        if exit_age:
            self.age_bins = self.age_bins[self.age_bins.age_group_start < exit_age]
            self.age_bins.loc[self.age_bins.age_group_end > exit_age, 'age_group_end'] = exit_age

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

        if self.by_year:
            total, born = self._metrics_by_year(pop, years, start_time, end_time, causes)

        else:
            alive_at_start = pop.loc[pop.entrance_time == self.initial_pop_entrance_time].copy()
            years_in_sim = to_years(alive_at_start.exit_time - alive_at_start.entrance_time)
            alive_at_start['age_at_year_start'] = alive_at_start.age - years_in_sim
            alive_at_start['age_at_year_end'] = alive_at_start.age

            born_in_sim = pop.loc[pop.entrance_time > self.initial_pop_entrance_time].copy()
            born_in_sim['age_at_year_end'] = born_in_sim.age

            total, born = self.count_deaths('all', alive_at_start, born_in_sim, self.age_bins, causes)

        for ((label, age_group), count) in total.unstack().iteritems():
            metrics[f'age_group_{age_group}_{label}'] = count

        for ((label, age_group), count) in born.unstack().iteritems():
            metrics[f'age_group_{age_group}_{label}_among_born'] = count

        return metrics

    def _metrics_by_year(self, pop, years, start_time, end_time, causes):
        total_data = []
        born_data = []

        for year in years:
            first_entrance = self.initial_pop_entrance_time if year == start_time.year else pd.datetime(year, 1, 1)
            last_exit = end_time if year == end_time.year else pd.datetime(year, 12, 31)

            existing_sims = pop.loc[(pop.entrance_time <= first_entrance) & (pop.exit_time > first_entrance)].copy()
            newborn_sims = pop.loc[(pop.entrance_time > first_entrance) & (pop.entrance_time < last_exit)].copy()

            max_years_in_this_year = to_years(existing_sims.exit_time - first_entrance)
            min_years_in_this_year = to_years(last_exit - first_entrance)

            existing_sims['age_at_year_start'] = existing_sims.age - max_years_in_this_year
            existing_sims['age_at_year_end'] = existing_sims.age_at_year_start \
                                               + np.minimum(max_years_in_this_year, min_years_in_this_year)

            max_newborn_years = to_years(last_exit - newborn_sims.entrance_time)
            newborn_sims['age_at_year_end'] = np.minimum(max_newborn_years, newborn_sims.age)

            total, born = self.count_deaths(year, existing_sims, newborn_sims, self.age_bins, causes)

            total_data.append(total)
            born_data.append(born)

        total = pd.concat(total_data)
        born = pd.concat(born_data)

        return total, born

    @staticmethod
    def count_deaths(year: Union[int, str], existing_sims: pd.DataFrame, newborn_sims: pd.DataFrame,
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
            in_group = existing_sims[(existing_sims.age >= start) & (existing_sims.age < end)]
            died = in_group[in_group.alive == 'dead']
            total.loc[group, f'total_deaths_{year}'] = len(died)
            cause_of_death = died.cause_of_death.value_counts()
            for cod, count in cause_of_death.iteritems():
                if 'death' not in cod:
                    cod = f'death_due_to_{cod}'
                cod += f'_{year}'
                total.loc[group, cod] += count

            in_group = newborn_sims[(newborn_sims.age >= start) & (newborn_sims.age < end)]
            died = in_group[in_group.alive == 'dead']
            born.loc[group, f'total_deaths_{year}'] = len(died)
            cause_of_death = died.cause_of_death.value_counts()
            for cod, count in cause_of_death.iteritems():
                if 'death' not in cod:
                    cod = f'death_due_to_{cod}'
                cod += f'_{year}'
                born.loc[group, cod] += count

            alive_at_start_and_lived_in = existing_sims[(existing_sims.age_at_year_start < end) &
                                                        (existing_sims.age_at_year_end >= start)]
            time_start = np.maximum(alive_at_start_and_lived_in.age_at_year_start, start)
            time_end = np.minimum(alive_at_start_and_lived_in.age_at_year_end, end)
            total.loc[group, f'person_time_{year}'] += (time_end - time_start).sum()

            born_and_lived_in = newborn_sims[(newborn_sims.age_at_year_end >= start) &
                                             (newborn_sims.age_at_year_end < end)]
            time_start = start
            time_end = np.minimum(born_and_lived_in.age_at_year_end, end)
            total.loc[group, f'person_time_{year}'] += (time_end - time_start).sum()

            lived_in = newborn_sims[(newborn_sims.age_at_year_end >= start) & (newborn_sims.age_at_year_end < end)]
            time_start = start
            time_end = np.minimum(lived_in.age_at_year_end, end)
            born.loc[group, f'person_time_{year}'] += (time_end - time_start).sum()

        return total, born

