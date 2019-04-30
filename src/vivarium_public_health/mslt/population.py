import numpy as np
import pandas as pd


class BasePopulation:
    """
    This component implements the core population demographics: age, sex,
    population size.

    The configuration options for this component are:

    - ``population_size``: The number of population cohorts (**must be
      specified**).

    - ``max_age``: The age at which cohorts are removed from the population
      (default: 110).

    .. code-block:: yaml

       configuration
           population:
               population_size: 44 # Male and female 5-year cohorts, 0 to 109.
               max_age: 110        # The age at which cohorts are removed.
    """

    configuration_defaults = {
        'population': {
            'max_age': 110,
        }
    }
    """Define the default age at which cohorts are removed."""

    def setup(self, builder):
        """Load the population data."""
        reqd_cols = ['age', 'sex', 'population', 'bau_population']
        zero_cols = ['acmr', 'bau_acmr',
                     'pr_death', 'bau_pr_death', 'deaths', 'bau_deaths',
                     'yld_rate', 'bau_yld_rate',
                     'person_years', 'bau_person_years',
                     'HALY', 'bau_HALY']

        self.pop_data = builder.data.load('population.structure')

        # Check that this table contains the required columns.
        present = set(reqd_cols) & set(self.pop_data.columns)
        if len(present) != len(reqd_cols):
            missing = set(reqd_cols) - set(self.pop_data.columns)
            msg = f'Table population.structure is missing columns: {missing}'
            raise ValueError(msg)

        # Create additional columns with placeholder (zero) values.
        for column in zero_cols:
            self.pop_data.loc[:, column] = 0.0

        self.max_age = builder.configuration.population.max_age

        # Track all of the quantities that exist in the core spreadsheet table.
        columns = reqd_cols + zero_cols
        builder.population.initializes_simulants(self.on_initialize_simulants, creates_columns=columns)
        self.population_view = builder.population.get_view(columns + ['tracked'])

        builder.event.register_listener('time_step', self.on_time_step)

    def on_initialize_simulants(self, _):
        """Initialize each cohort."""
        self.population_view.update(self.pop_data)

    def on_time_step(self, event):
        """Remove cohorts that have reached the maximum age."""
        pop = self.population_view.get(event.index, query='tracked == True')
        pop['age'] += 1
        pop.loc[pop.age > self.max_age, 'tracked'] = False
        self.population_view.update(pop)


class Mortality:
    """
    This component reduces the population size of each cohort over time,
    according to the all-cause mortality rate.
    """

    def setup(self, builder):
        """Load the all-cause mortality rate."""
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
        """
        Calculate the number of deaths and survivors at each time-step, for
        both the BAU and intervention scenarios.
        """
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
    """
    This component calculates the health-adjusted life years (HALYs) for each
    cohort over time, according to the years lost due to disability (YLD)
    rate.
    """

    def setup(self, builder):
        """Load the years lost due to disability (YLD) rate."""
        yld_data = builder.data.load('cause.all_causes.disability_rate')
        yld_rate = builder.lookup.build_table(yld_data)
        self.yld_rate = builder.value.register_rate_producer('yld_rate', source=yld_rate)

        builder.event.register_listener('time_step', self.on_time_step)

        self.population_view = builder.population.get_view([
            'bau_yld_rate', 'yld_rate',
            'bau_person_years', 'person_years',
            'bau_HALY', 'HALY'])

    def on_time_step(self, event):
        """
        Calculate the HALYs for each cohort at each time-step, for both the
        BAU and intervention scenarios.
        """
        pop = self.population_view.get(event.index)
        if pop.empty:
            return
        pop.yld_rate = self.yld_rate(event.index)
        pop.bau_yld_rate = self.yld_rate.source(event.index)
        pop.HALY = pop.person_years * (1 - pop.yld_rate)
        pop.bau_HALY = pop.bau_person_years * (1 - pop.bau_yld_rate)
        self.population_view.update(pop)
