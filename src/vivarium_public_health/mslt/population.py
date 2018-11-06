import numpy as np
import pandas as pd

from . import add_year_column


class BasePopulation:

    configuration_defaults = {
        'population': {
            'max_age': 110,
        }
    }

    def setup(self, builder):
        self.pop_data = builder.data.load('population.structure')
        self.max_age = builder.configuration.population.max_age

        columns = ['age', 'sex', 'population', 'bau_population']
        builder.population.initializes_simulants(self.on_initialize_simulants, creates_columns=columns)
        self.population_view = builder.population.get_view(columns + ['tracked'])

        builder.event.register_listener('time_step', self.on_time_step, priority=6)

    def on_initialize_simulants(self, _):
        self.population_view.update(self.pop_data)

    def on_time_step(self, event):
        pop = self.population_view.get(event.index, query='tracked == True')
        pop['age'] += 1
        pop.loc[pop.age >= self.max_age, 'tracked'] = False
        self.population_view.update(pop)


class Mortality:

    def setup(self, builder):
        mortality_data = builder.data.load('cause.all_causes.mortality')
        mortality_data = add_year_column(builder, mortality_data)
        self.mortality_rate = builder.value.register_rate_producer(
            'mortality_rate', source=builder.lookup.build_table(mortality_data))

        builder.event.register_listener('time_step', self.on_time_step)

        self.population_view = builder.population.get_view(['population', 'bau_population'])

    def on_time_step(self, event):
        pop = self.population_view.get(event.index)
        probability_of_death = 1 - np.exp(-self.mortality_rate(event.index))
        pop.population *= 1 - probability_of_death
        bau_probability_of_death = 1 - np.exp(-self.mortality_rate.source(event.index))
        pop.bau_population *= 1 - bau_probability_of_death
        self.population_view.update(pop)


class Disability:

    def setup(self, builder):
        yld_data = builder.data.load('cause.all_causes.disability_rate')
        yld_data = add_year_column(builder, yld_data)
        yld_rate = builder.lookup.build_table(yld_data)
        builder.value.register_rate_producer('yld_rate', source=yld_rate)
