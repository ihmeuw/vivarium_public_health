import pandas as pd

from vivarium.framework.utilities import rate_to_probability
from vivarium.framework.values import list_combiner

from .data_transformations import get_cause_deleted_mortality


class Mortality:

    def setup(self, builder):
        self._all_cause_mortality_data = builder.data.load("cause.all_causes.cause_specific_mortality")
        self._cause_deleted_mortality_data = None

        self._root_location = builder.configuration.input_data.location
        self._build_lookup_handle = builder.lookup.build_table

        self.csmr = builder.value.register_value_producer('csmr_data', source=list, preferred_combiner=list_combiner)
        self.mortality_rate = builder.value.register_rate_producer('mortality_rate', source=self.mortality_rate_source)

        life_expectancy_data = builder.data.load("population.theoretical_minimum_risk_life_expectancy")
        self.life_expectancy = builder.lookup.build_table(life_expectancy_data, key_columns=[],
                                                          parameter_columns=[('age','age_group_start', 'age_group_end')])

        self.death_emitter = builder.event.get_emitter('deaths')
        self.random = builder.randomness.get_stream('mortality_handler')
        self.clock = builder.time.clock()
        builder.value.register_value_modifier('metrics', modifier=self.metrics)

        self.population_view = builder.population.get_view(
            ['cause_of_death', 'alive', 'exit_time', 'age', 'sex', 'location', 'years_of_life_lost'])
        builder.population.initializes_simulants(self.load_population_columns,
                                                 creates_columns=['cause_of_death', 'years_of_life_lost'])
        builder.event.register_listener('time_step', self.mortality_handler, priority=0)

    def mortality_rate_source(self, index):
        if self._cause_deleted_mortality_data is None:
            csmr_data = self.csmr()
            cause_deleted_mr = get_cause_deleted_mortality(self._all_cause_mortality_data, csmr_data)
            self._cause_deleted_mortality_data = self._build_lookup_handle(
                cause_deleted_mr)

        return self._cause_deleted_mortality_data(index)

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

            self.death_emitter(event.split(dead_pop.index))

            self.population_view.update(dead_pop[['alive', 'exit_time', 'cause_of_death', 'years_of_life_lost']])

    def metrics(self, index, metrics):
        population = self.population_view.get(index)
        the_living = population[population.alive == 'alive']
        the_dead = population[population.alive == 'dead']
        metrics['years_of_life_lost'] = self.life_expectancy(the_dead.index).sum()
        metrics['total_population__living'] = len(the_living)
        metrics['total_population__dead'] = len(the_dead)

        for (condition, count) in pd.value_counts(the_dead.cause_of_death).to_dict().items():
            metrics['death_due_to_{}'.format(condition)] = count

        return metrics
