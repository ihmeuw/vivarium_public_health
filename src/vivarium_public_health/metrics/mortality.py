from .utilities import get_age_bins, get_person_time, get_deaths


class MortalityObserver:
    """ An observer for total and cause specific deaths during simulation.
    This component counts total and cause specific deaths in the population
    as well as person time (the time spent alive and tracked in the
    simulation).
    The data is discretized by age groups and optionally by year.
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
        deaths = get_deaths(pop, self.config.to_dict(), self.start_time, self.clock(), self.age_bins)

        metrics.update(person_time)
        metrics.update(deaths)

        return metrics

