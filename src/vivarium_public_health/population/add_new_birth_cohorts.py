"""
================
Fertility Models
================

This module contains several different models of fertility.

"""
import pandas as pd
import numpy as np

from vivarium_public_health import utilities
from vivarium_public_health.population.data_transformations import get_live_births_per_year

# TODO: Incorporate better data into gestational model (probably as a separate component)
PREGNANCY_DURATION = pd.Timedelta(days=9*utilities.DAYS_PER_MONTH)


class FertilityDeterministic:
    """Deterministic model of births."""

    configuration_defaults = {
        'fertility': {
            'number_of_new_simulants_each_year': 1000,
        },
    }

    @property
    def name(self):
        return "deterministic_fertility"

    def setup(self, builder):
        self.fractional_new_births = 0
        self.simulants_per_year = builder.configuration.fertility.number_of_new_simulants_each_year

        builder.population.register_simulant_creator('new_births', self.create_new_births)

    def create_new_births(self, simulant_creator, pop_data):
        """Adds a set number of simulants to the population each time step."""
        # Assume births are uniformly distributed throughout the year.
        step_size = utilities.to_years(pop_data.creation_window)
        pop_data.update({'age_start': 0, 'age_end': 0})

        simulants_to_add = self.simulants_per_year*step_size + self.fractional_new_births

        self.fractional_new_births = simulants_to_add % 1
        simulants_to_add = int(simulants_to_add)

        if simulants_to_add > 0:
            simulant_creator(simulants_to_add, pop_data)

    def __repr__(self):
        return "FertilityDeterministic()"


class FertilityCrudeBirthRate:
    """Population-level model of births using crude birth rate.

    The number of births added each time step is calculated as

    new_births = sim_pop_size_t0 * live_births / true_pop_size * step_size

    Where

    sim_pop_size_t0 = the initial simulation population size
    live_births = annual number of live births in the true population
    true_pop_size = the true population size

    This component has configuration flags that determine whether the
    live births and the true population size should vary with time.

    Notes
    -----
    The OECD definition of crude birth rate can be found on their
    `website <https://stats.oecd.org/glossary/detail.asp?ID=490>`_,
    while a more thorough discussion of fertility and
    birth rate models can be found on
    `Wikipedia <https://en.wikipedia.org/wiki/Birth_rate>`_ or in demography
    textbooks.

    """

    configuration_defaults = {
        'fertility': {
            'time_dependent_live_births': True,
            'time_dependent_population_fraction': False,
        }
    }

    @property
    def name(self):
        return "crude_birthrate_fertility"

    def setup(self, builder):
        self.birth_rate = get_live_births_per_year(builder)

        self.clock = builder.time.clock()
        self.randomness = builder.randomness.get_stream('crude_birth_rate')
        builder.population.register_simulant_creator('new_births', self.create_new_births)

    def create_new_births(self, simulant_creator, pop_data):
        """Adds new simulants every time step based on the Crude Birth Rate
        and an assumption that birth is a Poisson process

        """
        birth_rate = self.birth_rate.at[self.clock().year]
        step_size = utilities.to_years(pop_data.creation_window)

        mean_births = birth_rate * step_size
        # Assume births occur as a Poisson process
        r = np.random.RandomState(seed=self.randomness.get_seed())
        simulants_to_add = r.poisson(mean_births)
        pop_data.update({'age_start': 0, 'age_end': 0})

        if simulants_to_add > 0:
            simulant_creator(simulants_to_add, pop_data)

    def __repr__(self):
        return "FertilityCrudeBirthRate()"


class FertilityAgeSpecificRates:
    """
    A simulant-specific model for fertility and pregnancies.
    """

    @property
    def name(self):
        return 'age_specific_fertility'

    def setup(self, builder):
        """ Setup the common randomness stream and
        age-specific fertility lookup tables.
        Parameters
        ----------
        builder : vivarium.engine.Builder
            Framework coordination object.
        """
        age_specific_fertility_rate = self.load_age_specific_fertility_rate_data(builder)
        fertility_rate = builder.lookup.build_table(age_specific_fertility_rate, parameter_columns=['age', 'year'])
        self.fertility_rate = builder.value.register_rate_producer('fertility rate',
                                                                   source=fertility_rate,
                                                                   requires_columns=['age'])

        self.randomness = builder.randomness.get_stream('fertility')

        self.population_view = builder.population.get_view(['last_birth_time', 'sex', 'parent_id'])
        builder.population.register_simulant_creator('new_births', self.create_new_births)

        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 creates_columns=['last_birth_time', 'parent_id'],
                                                 requires_columns=['sex'])

    def on_initialize_simulants(self, pop_data):
        """ Adds 'last_birth_time' and 'parent' columns to the state table."""
        pop = self.population_view.subview(['sex']).get(pop_data.index)
        women = pop.loc[pop.sex == 'Female'].index

        if pop_data.user_data['sim_state'] == 'setup':
            parent_id = -1
        else:  # 'sim_state' == 'time_step'
            parent_id = pop_data.user_data['parent_ids']
        pop_update = pd.DataFrame({'last_birth_time': pd.NaT, 'parent_id': parent_id}, index=pop_data.index)
        # FIXME: This is a misuse of the column and makes it invalid for
        #    tracking metrics.
        # Do the naive thing, set so all women can have children
        # and none of them have had a child in the last year.
        pop_update.loc[women, 'last_birth_time'] = pop_data.creation_time - pd.Timedelta(days=utilities.DAYS_PER_YEAR)

        self.population_view.update(pop_update)

    def create_new_births(self, simulant_creator, pop_data):
        """Produces new children and updates parent status on time steps."""
        # Get a view on all living women who haven't had a child in at least nine months.
        nine_months_ago = pd.Timestamp(pop_data.creation_time - PREGNANCY_DURATION)
        population = self.population_view.get(pop_data.existing_index, query='alive == "alive" and sex =="Female"')
        can_have_children = population.last_birth_time < nine_months_ago
        eligible_women = population[can_have_children]

        rate_series = self.fertility_rate(eligible_women.index)
        had_children = self.randomness.filter_for_rate(eligible_women, rate_series).copy()

        had_children.loc[:, 'last_birth_time'] = pop_data.creation_time
        self.population_view.update(had_children['last_birth_time'])

        # If children were born, add them to the state table and record
        # who their mother was.
        num_babies = len(had_children)
        pop_data.update({'age_start': 0, 'age_end': 0, 'parent_ids': had_children.index})
        if num_babies:
            simulant_creator(num_babies, pop_data)

    def load_age_specific_fertility_rate_data(self, builder):
        asfr_data = builder.data.load("covariate.age_specific_fertility_rate.estimate")
        columns = ['year_start', 'year_end', 'age_start', 'age_end', 'mean_value']
        asfr_data = asfr_data.loc[asfr_data.sex == 'Female'][columns]
        return asfr_data

    def __repr__(self):
        return "FertilityAgeSpecificRates()"
