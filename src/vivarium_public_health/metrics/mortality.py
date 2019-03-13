from .utilities import get_age_bins, get_person_time, get_deaths, get_years_of_life_lost


class MortalityObserver:
    """ An observer for cause-specific deaths, ylls, and total person time.

    The data is optionally discretized by age, sex, and/or year. These options
    can be configured in the model specification.
    """
    configuration_defaults = {
        'metrics': {
            'mortality': {
                'by_age': False,
                'by_year': False,
                'by_sex': False,
            }
        }
    }

    def setup(self, builder):
        self.name = 'mortality_observer'
        self.config = builder.configuration.metrics.mortality
        self.clock = builder.time.clock()
        self.step_size = builder.time.step_size()
        self.start_time = self.clock()
        self.initial_pop_entrance_time = self.start_time - self.step_size()
        self.age_bins = get_age_bins(builder)
        self.causes = [c.state_column for c in builder.components.get_components('DiseaseModel')]

        life_expectancy_data = builder.data.load("population.theoretical_minimum_risk_life_expectancy")
        self.life_expectancy = builder.lookup.build_table(life_expectancy_data, key_columns=[],
                                                          parameter_columns=[('age', 'age_group_start',
                                                                              'age_group_end')])

        columns_required = ['tracked', 'alive', 'entrance_time', 'exit_time', 'cause_of_death', 'years_of_life_lost']
        if self.config.by_age:
            columns_required += ['age']
        if self.config.by_sex:
            columns_required += ['sex']
        self.population_view = builder.population.get_view(columns_required)

        builder.value.register_value_modifier('metrics', self.metrics)

    def metrics(self, index, metrics):
        pop = self.population_view.get(index)
        pop.loc[pop.exit_time.isnull(), 'exit_time'] = self.clock()

        person_time = get_person_time(pop, self.config.to_dict(), self.start_time, self.clock(), self.age_bins)
        deaths = get_deaths(pop, self.config.to_dict(), self.start_time, self.clock(), self.age_bins, self.causes)
        ylls = get_years_of_life_lost(pop, self.config.to_dict(), self.start_time, self.clock(),
                                      self.age_bins, self.life_expectancy, self.causes)

        metrics.update(person_time)
        metrics.update(deaths)
        metrics.update(ylls)

        return metrics
