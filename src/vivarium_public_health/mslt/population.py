import numpy as np
import pandas as pd


class BasePopulation:

    configuration_defaults = {
        'population': {
            'max_age': 110,
        }
    }

    def setup(self, builder):
        self.pop_data = builder.data.load('population.structure')
        self.pop_data.loc[:, 'acmr'] = 0.0
        self.pop_data.loc[:, 'bau_acmr'] = 0.0
        self.pop_data.loc[:, 'pr_death'] = 0.0
        self.pop_data.loc[:, 'bau_pr_death'] = 0.0
        self.pop_data.loc[:, 'deaths'] = 0.0
        self.pop_data.loc[:, 'bau_deaths'] = 0.0
        self.pop_data.loc[:, 'yld_rate'] = 0.0
        self.pop_data.loc[:, 'bau_yld_rate'] = 0.0
        self.pop_data.loc[:, 'person_years'] = 0.0
        self.pop_data.loc[:, 'bau_person_years'] = 0.0
        self.pop_data.loc[:, 'HALY'] = 0.0
        self.pop_data.loc[:, 'bau_HALY'] = 0.0

        self.max_age = builder.configuration.population.max_age

        # Track all of the quantities that exist in the core spreadsheet table.
        columns = ['age', 'sex', 'population', 'bau_population',
                   'acmr', 'bau_acmr',
                   'pr_death', 'bau_pr_death', 'deaths', 'bau_deaths',
                   'yld_rate', 'bau_yld_rate',
                   'person_years', 'bau_person_years',
                   'HALY', 'bau_HALY']
        builder.population.initializes_simulants(self.on_initialize_simulants, creates_columns=columns)
        self.population_view = builder.population.get_view(columns + ['tracked'])

        builder.event.register_listener('time_step', self.on_time_step)

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
        self.mortality_rate = builder.value.register_rate_producer(
            'mortality_rate', source=builder.lookup.build_table(mortality_data))

        builder.event.register_listener('time_step', self.on_time_step)

        self.population_view = builder.population.get_view(['population', 'bau_population',
                                                            'acmr', 'bau_acmr',
                                                            'pr_death', 'bau_pr_death',
                                                            'deaths', 'bau_deaths',
                                                            'person_years', 'bau_person_years'])

    def on_time_step(self, event):
        pop = self.population_view.get(event.index)
        if pop.empty:
            return
        probability_of_death = 1 - np.exp(-self.mortality_rate(event.index))
        pop.acmr = self.mortality_rate(event.index)
        deaths = pop.population * probability_of_death
        pop.population *= 1 - probability_of_death
        bau_probability_of_death = 1 - np.exp(-self.mortality_rate.source(event.index))
        pop.bau_acmr = self.mortality_rate.source(event.index)
        bau_deaths = pop.bau_population * bau_probability_of_death
        pop.bau_population *= 1 - bau_probability_of_death
        pop.pr_death = probability_of_death
        pop.bau_pr_death = bau_probability_of_death
        pop.deaths = deaths
        pop.bau_deaths = bau_deaths
        pop.person_years = pop.population + 0.5 * pop.deaths
        pop.bau_person_years = pop.bau_population + 0.5 * pop.bau_deaths
        self.population_view.update(pop)


class Disability:

    def setup(self, builder):
        yld_data = builder.data.load('cause.all_causes.disability_rate')
        yld_rate = builder.lookup.build_table(yld_data)
        self.yld_rate = builder.value.register_rate_producer('yld_rate', source=yld_rate)

        builder.event.register_listener('time_step', self.on_time_step)

        self.population_view = builder.population.get_view([
            'bau_yld_rate', 'yld_rate',
            'bau_person_years', 'person_years',
            'bau_HALY', 'HALY'])

    def on_time_step(self, event):
        pop = self.population_view.get(event.index)
        if pop.empty:
            return
        pop.yld_rate = self.yld_rate(event.index)
        pop.bau_yld_rate = self.yld_rate.source(event.index)
        pop.HALY = pop.person_years * (1 - pop.yld_rate)
        pop.bau_HALY = pop.bau_person_years * (1 - pop.bau_yld_rate)
        self.population_view.update(pop)
