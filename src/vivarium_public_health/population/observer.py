import pandas as pd


def by_year(config):
    return ('observer' in config) and ('mortality' in config['observer']) and config['observer'].mortality.by_year


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

        modifier = self.metrics_by_year if by_year(builder.configuration) else self.metrics
        builder.value.register_value_modifier('metrics', modifier)

    def metrics(self, index, metrics):
        end_time = self.clock()
        pop = self.population_view.get(index)

        pop.loc[pop.exit_time.isnull(), 'exit_time'] = end_time
        alive_at_start = pop.loc[pop.entrance_time == self.initial_pop_entrance_time].copy()
        years_in_sim = (alive_at_start.exit_time - alive_at_start.entrance_time) / pd.Timedelta(days=365.25)
        alive_at_start['age_at_start'] = alive_at_start.age - years_in_sim
        born_in_sim = pop.loc[pop.entrance_time > self.initial_pop_entrance_time].copy()

        causes = pop.cause_of_death.unique().tolist()
        causes.remove('not_dead')
        causes.remove('death_due_to_other_causes')
        causes = [f'death_due_to_{c}' for c in causes]
        causes.append('death_due_to_other_causes')
        frame_dict = {'total_deaths': 0, 'person_time': 0}
        frame_dict.update({c: 0 for c in causes})

        total = pd.DataFrame(frame_dict, index=self.age_bins.index)
        born = pd.DataFrame(frame_dict, index=self.age_bins.index)

        for group, age_bin in self.age_bins.iterrows():
            start, end = age_bin.age_group_start, age_bin.age_group_end
            in_group = pop[(pop.age >= start) & (pop.age < end)]
            died = in_group[in_group.alive == 'dead']
            total.loc[group, 'total_deaths'] = len(died)
            cause_of_death = died.cause_of_death.value_counts()
            for cod, count in cause_of_death.iteritems():
                if 'death' not in cod:
                    cod = f'death_due_to_{cod}'
                total.loc[group, cod] += count

            in_group = born_in_sim[(born_in_sim.age >= start) & (born_in_sim.age < end)]
            died = in_group[in_group.alive == 'dead']
            born.loc[group, 'total_deaths'] = len(died)
            cause_of_death = died.cause_of_death.value_counts()
            for cod, count in cause_of_death.iteritems():
                if 'death' not in cod:
                    cod = f'death_due_to_{cod}'
                born.loc[group, cod] += count

            alive_at_start_and_lived_in = alive_at_start[(alive_at_start.age >= start) & (alive_at_start.age < end)]
            time_start = alive_at_start_and_lived_in.age_at_start
            time_start[time_start <= start] = start
            time_end = alive_at_start_and_lived_in.age
            time_end[time_end > end] = end
            total.loc[group, 'person_time'] += (time_end - time_start).sum()

            born_and_lived_in = born_in_sim[born_in_sim.age >= start]
            time_start = start
            time_end = born_and_lived_in.age
            time_end[time_end > end] = end
            total.loc[group, 'person_time'] += (time_end - time_start).sum()

            lived_in = born_in_sim[born_in_sim.age >= start]
            time_start = start
            time_end = lived_in.age
            time_end[time_end > end] = end
            born.loc[group, 'person_time'] += (time_end - time_start).sum()

        for ((label, age_group), count) in total.unstack().iteritems():
            metrics[f'age_group_{age_group}_{label}'] = count

        for ((label, age_group), count) in born.unstack().iteritems():
            metrics[f'age_group_{age_group}_{label}_among_born'] = count

        return metrics

    def metrics_by_year(self, index, metrics):
        start_time = self.start_time
        end_time = self.clock()
        pop = self.population_view.get(index)

        pop.loc[pop.exit_time.isnull(), 'exit_time'] = end_time
        alive_at_start = pop.loc[pop.entrance_time == self.initial_pop_entrance_time].copy()
        years_in_sim = (alive_at_start.exit_time - alive_at_start.entrance_time) / pd.Timedelta(days=365.25)
        alive_at_start['age_at_start'] = alive_at_start.age - years_in_sim
        born_in_sim = pop.loc[pop.entrance_time > self.initial_pop_entrance_time].copy()

        causes = pop.cause_of_death.unique().tolist()
        causes.remove('not_dead')
        causes.remove('death_due_to_other_causes')
        causes = [f'death_due_to_{c}' for c in causes]
        causes.append('death_due_to_other_causes')
        frame_dict = {'total_deaths': 0, 'person_time': 0}
        frame_dict.update({c: 0 for c in causes})

        total = pd.DataFrame(frame_dict, index=self.age_bins.index)
        born = pd.DataFrame(frame_dict, index=self.age_bins.index)

        # let's leave out the first year and the last year

        for year in range(start_time.year+1, end_time.year):
            existing_sims = pop.loc[(pop.entrance_time.dt.year < year) & (pop.exit_time.dt.year >= year)].copy()
            newborn_sims = pop.loc[pop.entrance_time.dt.year == year].copy()

            sim_years_since_year_start = (existing_sims.exit_time - pd.datetime(year, 1, 1)) / pd.Timedelta(days=365.25)
            existing_sims['age_at_year_start'] = existing_sims.age - sim_years_since_year_start
            existing_sims['age_at_year_end'] = existing_sims.age_at_year_start + min(1, sim_years_since_year_start)

            sim_years_till_year_end = (pd.datetime(year, 12, 31) - newborn_sims.entrace_time) / pd.Timedelta(days=365.25)
            newborn_sims['age_at_year_end'] = min(sim_years_till_year_end, newborn_sims.age)

            for group, age_bin in self.age_bins.iterrows():
                start, end = age_bin.age_group_start, age_bin.age_group_end
                in_group = existing_sims[(existing_sims.age >= start) & (existing_sims.age < end)]
                died = in_group[in_group.alive == 'dead']
                total.loc[group, f'total_deaths_{year}'] = len(died)
                cause_of_death = died.cause_of_death.value_counts()
                for cod, count in cause_of_death.iteritems():
                    if 'death' not in cod:
                        cod = f'death_due_to_{cod}_{year}'
                    total.loc[group, cod] += count

                in_group = newborn_sims[(newborn_sims.age >= start) & (newborn_sims.age < end)]
                died = in_group[in_group.alive == 'dead']
                born.loc[group, f'total_deaths_{year}'] = len(died)
                cause_of_death = died.cause_of_death.value_counts()
                for cod, count in cause_of_death.iteritems():
                    if 'death' not in cod:
                        cod = f'death_due_to_{cod}_{year}'
                    born.loc[group, cod] += count

                alive_at_start_and_lived_in = existing_sims[(existing_sims.age_at_year_start < end) &
                                                            (existing_sims.age_at_year_end >= start)]
                time_start = max(alive_at_start_and_lived_in.age_at_year_start, start)
                time_end = min(alive_at_start_and_lived_in.age_at_year_end, end)
                total.loc[group, f'person_time_{year}'] += (time_end - time_start).sum()

                born_and_lived_in = newborn_sims[(newborn_sims.age_at_year_end >= start) &(newborn_sims.age_at_year_end < end)]
                time_start = start
                time_end = min(born_and_lived_in.age_at_year_end, end)
                total.loc[group, f'person_time_{year}'] += (time_end - time_start).sum()

                lived_in = newborn_sims[(newborn_sims.age_at_year_end >= start) &(newborn_sims.age_at_year_end < end)]
                time_start = start
                time_end = min(lived_in.age_at_year_end, end)
                born.loc[group, f'person_time_{year}'] += (time_end - time_start).sum()

            for ((label, age_group), count) in total.unstack().iteritems():
                metrics[f'age_group_{age_group}_{label}'] = count

            for ((label, age_group), count) in born.unstack().iteritems():
                metrics[f'age_group_{age_group}_{label}_among_born'] = count

            return metrics

