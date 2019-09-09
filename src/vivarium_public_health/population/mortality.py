"""
========================
The Core Mortality Model
========================

This module contains tools modeling all cause mortality and hooks for
disease models to contribute cause-specific and excess mortality.

"""
import pandas as pd

from vivarium.framework.utilities import rate_to_probability


class Mortality:

    @property
    def name(self):
        return 'mortality'

    def setup(self, builder):
        all_cause_mortality_data = builder.data.load("cause.all_causes.cause_specific_mortality_rate")
        self.all_cause_mortality_rate = builder.lookup.build_table(all_cause_mortality_data)
        self.cause_specific_mortality_rate = builder.value.register_value_producer(
            'cause_specific_mortality_rate', source=builder.lookup.build_table(0)
        )

        self.mortality_rate = builder.value.register_rate_producer('mortality_rate',
                                                                   source=self._mortality_rate)

        life_expectancy_data = builder.data.load("population.theoretical_minimum_risk_life_expectancy")
        self.life_expectancy = builder.lookup.build_table(life_expectancy_data, key_columns=[],
                                                          parameter_columns=[('age', 'age_start', 'age_end')])

        self.random = builder.randomness.get_stream('mortality_handler')
        self.clock = builder.time.clock()

        self.population_view = builder.population.get_view(
            ['cause_of_death', 'alive', 'exit_time', 'age', 'sex', 'location', 'years_of_life_lost'])
        builder.population.initializes_simulants(self.load_population_columns,
                                                 creates_columns=['cause_of_death', 'years_of_life_lost'])
        builder.event.register_listener('time_step', self.mortality_handler, priority=0)

    def _mortality_rate(self, index):
        acmr = self.all_cause_mortality_rate(index)
        csmr = self.cause_specific_mortality_rate(index, skip_post_processor=True)
        return acmr - csmr

    def load_population_columns(self, pop_data):
        self.population_view.update(pd.DataFrame({'cause_of_death': 'not_dead',
                                                  'years_of_life_lost': 0.}, index=pop_data.index))

    def mortality_handler(self, event):
        pop = self.population_view.get(event.index, query="alive =='alive'")
        prob_df = rate_to_probability(pd.DataFrame(self.mortality_rate(pop.index)))
        prob_df['no_death'] = 1-prob_df.sum(axis=1)
        prob_df['cause_of_death'] = self.random.choice(prob_df.index, prob_df.columns, prob_df)
        dead_pop = prob_df.query('cause_of_death != "no_death"').copy()

        if not dead_pop.empty:
            dead_pop['alive'] = pd.Series('dead', index=dead_pop.index)
            dead_pop['exit_time'] = event.time
            dead_pop['years_of_life_lost'] = self.life_expectancy(dead_pop.index)
            self.population_view.update(dead_pop[['alive', 'exit_time', 'cause_of_death', 'years_of_life_lost']])

    def __repr__(self):
        return "Mortality()"
