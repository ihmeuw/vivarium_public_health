import pandas as pd


class MortalityObserver:

    def setup(self, builder):
        self.clock = builder.time.clock()
        self.step_size = builder.time.step_size()
        self.initial_pop_entrance_time = self.clock() - self.step_size()
        columns_required = ['tracked', 'alive', 'age', 'entrance_time', 'exit_time',
                            'cause_of_death', 'years_of_life_lost']
        self.age_bins = (builder.data.load('population.demographic_dimensions')[['age_group_start', 'age_group_end']]
                         .drop_duplicates()
                         .reset_index(drop=True))
        if builder.configuration.population.exit_age:
            self.age_bins = self.age_bins[self.age_bins.age_group_end <= builder.configuration.population.exit_age]

        self.population_view = builder.population.get_view(columns_required)
        builder.value.register_value_modifier('metrics', self.metrics)

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
            in_group = pop[pop.age.between(start, end)]
            died = in_group[in_group.alive == 'dead']
            total.loc[group, 'total_deaths'] = len(died)
            cause_of_death = died.cause_of_death.value_counts()
            for cod, count in cause_of_death.iteritems():
                if 'death' not in cod:
                    cod = f'death_due_to_{cod}'
                total.loc[group, cod] += count

            in_group = born_in_sim[born_in_sim.age.between(start, end)]
            died = in_group[in_group.alive == 'dead']
            born.loc[group, 'total_deaths'] = len(died)
            cause_of_death = died.cause_of_death.value_counts()
            for cod, count in cause_of_death.iteritems():
                if 'death' not in cod:
                    cod = f'death_due_to_{cod}'
                born.loc[group, cod] += count

            alive_at_start_and_lived_in = alive_at_start[alive_at_start.age_at_start.between(start, end)]
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
