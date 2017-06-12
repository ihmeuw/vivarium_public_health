from datetime import timedelta

import numpy as np
import pandas as pd

from ceam_inputs import get_life_table, get_cause_deleted_mortality_rate

from ceam import config
from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns
from ceam.framework.util import rate_to_probability
from ceam.framework.values import list_combiner, produces_value, modifies_value

from ceam_public_health.util import make_cols_demographically_specific

susceptible_person_time_cols = make_cols_demographically_specific("susceptible_person_time", 2, 5)
diarrhea_event_count_cols = make_cols_demographically_specific("diarrhea_event_count", 2, 5)


class Mortality:
    def setup(self, builder):
        self._mortality_rate_builder = lambda: builder.lookup(self.load_all_cause_mortality())
        self.mortality_rate = builder.rate('mortality_rate')
        self.death_emitter = builder.emitter('deaths')
        self.life_table = builder.lookup(get_life_table(), key_columns=(), parameter_columns=('age',))
        self.random = builder.randomness('mortality_handler')
        self.csmr_data = builder.value('csmr_data', list_combiner)
        self.csmr_data.source = list
        self.clock = builder.clock()

    @listens_for('post_setup')
    def post_step(self, event):
        # This is being loaded after the main setup phase because it needs to happen after all disease models
        # have completed their setup phase which isn't guaranteed (or even likely) during this component's
        # normal setup.
        self.mortality_rate_lookup = self._mortality_rate_builder()

    def load_all_cause_mortality(self):
        return get_cause_deleted_mortality_rate(self.csmr_data())

    @listens_for('initialize_simulants')
    @uses_columns(['death_day', 'cause_of_death'])
    def death_day_column(self, event):
        event.population_view.update(pd.Series(pd.NaT, name='death_day', index=event.index))
        event.population_view.update(pd.Series('not_dead', name='cause_of_death', index=event.index))

    # FIXME: Set the time of death to be the midpoint between the current and next time step. this is important for the mortality rate calculations
    @listens_for('time_step', priority=0)
    @uses_columns(['alive', 'death_day', 'cause_of_death'], 'alive')
    def mortality_handler(self, event):
        rate_df = self.mortality_rate(event.index)

        # make sure to turn the rates into probabilities, do a cumulative sum to make sure that people can only die from one cause
        # first convert to probabilities
        prob_df = rate_to_probability(rate_df)

        # determine if simulant has died, assign cause of death
        prob_df['no_death'] = 1-prob_df.sum(axis=1)

        prob_df['cause_of_death'] = self.random.choice(prob_df.index, prob_df.columns, prob_df)

        dead_pop = prob_df.query('cause_of_death != "no_death"').copy()
        sub = dead_pop.query('cause_of_death != "death_due_to_other_causes"')

        dead_pop['alive'] = False

        self.death_emitter(event.split(dead_pop.index))

        dead_pop['death_day'] = event.time

        event.population_view.update(dead_pop[['alive', 'death_day', 'cause_of_death']])

    @produces_value('mortality_rate')
    def mortality_rate_source(self, population):
        return pd.DataFrame({'death_due_to_other_causes': self.mortality_rate_lookup(population)})

    @modifies_value('metrics')
    @uses_columns(['alive', 'age', 'cause_of_death'])
    def metrics(self, index, metrics, population_view):
        population = population_view.get(index)
        the_dead = population.query('not alive')
        metrics['deaths'] = len(the_dead)
        metrics['years_of_life_lost'] = self.life_table(the_dead.index).sum()
        metrics['total_population'] = len(population)
        metrics['total_population__living'] = len(population) - len(the_dead)
        metrics['total_population__dead'] = len(the_dead)
        for (condition, count) in pd.value_counts(the_dead.cause_of_death).to_dict().items():
            metrics['{}'.format(condition)] = count

        return metrics

    @modifies_value('epidemiological_span_measures')
    @uses_columns(['age', 'death_day', 'cause_of_death', 'alive', 'sex'])
    def calculate_mortality_measure(self, index, age_groups, sexes, all_locations, duration, cube, population_view):
        root_location = config.simulation_parameters.location_id
        pop = population_view.get(index)

        if all_locations:
            locations = set(pop.location) | {-1}
        else:
            locations = {-1}

        now = self.clock()
        window_start = now - duration

        causes_of_death = set(pop.cause_of_death.unique()) - {'not_dead'}

        for low, high in age_groups:
            for sex in sexes:
                for location in locations:
                    sub_pop = pop.query('age > @low and age <= @high and sex == @sex and (alive or death_day > @window_start)')
                    if location >= 0:
                        sub_pop = sub_pop.query('location == @location')

                    if not sub_pop.empty:
                        birthday = sub_pop.death_day.fillna(now) - pd.to_timedelta(sub_pop.age, 'Y')
                        time_before_birth = np.maximum(np.timedelta64(0), birthday - window_start).dt.total_seconds().sum()
                        time_after_death = np.minimum(np.maximum(np.timedelta64(0), now - sub_pop.death_day.dropna()), np.timedelta64(duration)).dt.total_seconds().sum()
                        time_in_sim = duration.total_seconds() * len(pop) - (time_before_birth + time_after_death)
                        time_in_sim = time_in_sim/(timedelta(days=364).total_seconds())
                        for cause in causes_of_death:
                            deaths_in_period = (sub_pop.cause_of_death == cause).sum()

                            cube = cube.append(pd.DataFrame({'measure': 'mortality', 'age_low': low, 'age_high': high, 'sex': sex, 'location': location if location >= 0 else root_location, 'cause': cause, 'value': deaths_in_period/time_in_sim, 'sample_size': len(sub_pop)}, index=[0]).set_index(['measure', 'age_low', 'age_high', 'sex', 'location', 'cause']))
                        deaths_in_period = len(sub_pop.query('not alive'))
                        cube = cube.append(pd.DataFrame({'measure': 'mortality', 'age_low': low, 'age_high': high, 'sex': sex, 'location': location if location >= 0 else root_location, 'cause': 'all', 'value': deaths_in_period/time_in_sim, 'sample_size': len(sub_pop)}, index=[0]).set_index(['measure', 'age_low', 'age_high', 'sex', 'location', 'cause']))
        return cube

    @modifies_value('epidemiological_span_measures')
    @uses_columns(['death_day', 'sex', 'age', 'location'], 'not alive')
    def deaths(self, index, age_groups, sexes, all_locations, duration, cube, population_view):
        root_location = config.simulation_parameters.location_id
        pop = population_view.get(index)

        if all_locations:
            locations = set(pop.location) | {-1}
        else:
            locations = {-1}

        now = self.clock()
        window_start = now - duration
        for low, high in age_groups:
            for sex in sexes:
                for location in locations:
                    sub_pop = pop.query('age > @low and age <= @high and sex == @sex')
                    sample_size = len(sub_pop)
                    sub_pop = sub_pop.query('death_day > @window_start and death_day <= @now')
                    if location >= 0:
                        sub_pop = sub_pop.query('location == @location')

                    cube = cube.append(pd.DataFrame({'measure': 'deaths', 'age_low': low, 'age_high': high, 'sex': sex, 'location': location if location >= 0 else root_location, 'cause': 'all', 'value': len(sub_pop), 'sample_size': sample_size}, index=[0]).set_index(['measure', 'age_low', 'age_high', 'sex', 'location', 'cause']))
        return cube

    # TODO: Would be nice to use age_group_name instead of age_group_high and age_group_low. Using age_group_name is more specific, will make the graphs cleaner, and is more interpretable for the under 1 (neonatal) age groups.
    # FIXME: Should move the epi measures code to its own class, probably its own script
    @modifies_value('epidemiological_span_measures')
    @uses_columns(['age', 'death_day', 'cause_of_death', 'alive', 'sex'] + susceptible_person_time_cols + diarrhea_event_count_cols)
    def calculate_incidence_measure(self, index, age_groups, sexes, all_locations, duration, cube, population_view):
        root_location = config.getint('simulation_parameters', 'location_id')
        pop = population_view.get(index)

        if all_locations:
            locations = set(pop.location) | {-1}
        else:
            locations = {-1}

        now = self.clock()
        window_start = now - duration
        current_year = window_start.year


        # FIXME: Don't want to have age_groups[0:3] hard-coded in. Need to make a component that calculates susceptible person time for all age groups so that this can be avoided
        for low, high in age_groups[0:4]:
            for sex in sexes:
                for location in locations:
                    sub_pop = pop.query('age > @low and age <= @high and sex == @sex')
                    low_str = str(np.round(low, 2))
                    high_str = str(np.round(high, 2))
                    if location >= 0:
                        sub_pop = sub_pop.query('location == @location')

                    # TODO: Make this more flexible. Don't want to have diarrhea hard-coded in here. Want the susceptibility column and disease column to be variables that get passed into the class.
                    # TODO: Need to figure out best place for this
                    if not sub_pop.empty:
                        susceptible_person_time = pop["susceptible_person_time_{l}_to_{h}_in_year_{y}_among_{s}s".format(l=low_str, h=high_str, y=current_year, s=sex)].sum()
                        num_diarrhea_cases = pop['diarrhea_event_count_{l}_to_{h}_in_year_{y}_among_{s}s'.format(l=low_str, h=high_str, y=current_year, s=sex)].sum()
                        if susceptible_person_time != 0:
                            cube = cube.append(pd.DataFrame({'measure': 'incidence', 'age_low': low, 'age_high': high, 'sex': sex, 'location': location if location >= 0 else root_location, 'cause': 'diarrhea', 'value': num_diarrhea_cases/susceptible_person_time, 'sample_size': len(sub_pop)}, index=[0]).set_index(['measure', 'age_low', 'age_high', 'sex', 'location', 'cause']))
        return cube