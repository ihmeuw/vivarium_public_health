from vivarium_public_health.disease import DiseaseModel
from .utilities import get_age_bins, get_person_time, get_deaths, get_years_of_life_lost


class MortalityObserver:
    """ An observer for cause-specific deaths, ylls, and total person time.

    By default, this counts cause-specific deaths, years of life lost, and
    total person time over the full course of the simulation. It can be
    configured to bin these measures into age groups, sexes, and years
    by setting the ``by_age``, ``by_sex``, and ``by_year`` flags, respectively.

    In the model specification, your configuration for this component should
    be specified as, e.g.:

    .. code-block:: yaml

        configuration:
            metrics:
                mortality:
                    by_age: True
                    by_year: False
                    by_sex: True

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

    def __init__(self):
        self.name = 'mortality_observer'

    def setup(self, builder):
        self.config = builder.configuration.metrics.mortality
        self.clock = builder.time.clock()
        self.step_size = builder.time.step_size()
        self.start_time = self.clock()
        self.initial_pop_entrance_time = self.start_time - self.step_size()
        self.age_bins = get_age_bins(builder)
        self.causes = [c.state_column for c in builder.components.get_components(DiseaseModel)] + ['other_causes']

        life_expectancy_data = builder.data.load("population.theoretical_minimum_risk_life_expectancy")
        self.life_expectancy = builder.lookup.build_table(life_expectancy_data, key_columns=[],
                                                          parameter_columns=[('age', 'age_group_start',
                                                                              'age_group_end')])

        columns_required = ['tracked', 'alive', 'entrance_time', 'exit_time', 'cause_of_death',
                            'years_of_life_lost', 'age']
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

        the_living = pop[(pop.alive == 'alive') & pop.tracked]
        the_dead = pop[pop.alive == 'dead']
        metrics['years_of_life_lost'] = self.life_expectancy(the_dead.index).sum()
        metrics['total_population_living'] = len(the_living)
        metrics['total_population_dead'] = len(the_dead)

        return metrics
