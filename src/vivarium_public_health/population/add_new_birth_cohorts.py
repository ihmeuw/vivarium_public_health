"""
================
Fertility Models
================

This module contains several different models of fertility.

"""
import pandas as pd
import numpy as np
from vivarium_public_health.population.data_transformations import get_live_births_per_year

SECONDS_PER_YEAR = 365.25*24*60*60
# TODO: Incorporate better data into gestational model (probably as a separate component)
PREGNANCY_DURATION = pd.Timedelta(days=9*30.5)


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

        self.simulant_creator = builder.population.get_simulant_creator()

        builder.event.register_listener('time_step', self.on_time_step)

    def on_time_step(self, event):
        """Adds a set number of simulants to the population each time step.

        Parameters
        ----------
        event
            The event that triggered the function call.
        """
        # Assume births are uniformly distributed throughout the year.
        step_size = event.step_size/pd.Timedelta(seconds=SECONDS_PER_YEAR)
        simulants_to_add = self.simulants_per_year*step_size + self.fractional_new_births

        self.fractional_new_births = simulants_to_add % 1
        simulants_to_add = int(simulants_to_add)

        if simulants_to_add > 0:
            self.simulant_creator(simulants_to_add,
                                  population_configuration={
                                      'age_start': 0,
                                      'age_end': 0,
                                      'sim_state': 'time_step',
                                  })

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
        self.clock = builder.time.clock()

        self.birth_rate = get_live_births_per_year(builder)

        self.randomness = builder.randomness.get_stream('crude_birth_rate')

        self.simulant_creator = builder.population.get_simulant_creator()

        builder.event.register_listener('time_step', self.on_time_step)

    def on_time_step(self, event):
        """Adds new simulants every time step based on the Crude Birth Rate
        and an assumption that birth is a Poisson process
        Parameters
        ----------
        event
            The event that triggered the function call.
        """
        birth_rate = self.birth_rate.at[self.clock().year]
        step_size = event.step_size / pd.Timedelta(seconds=SECONDS_PER_YEAR)

        mean_births = birth_rate * step_size
        # Assume births occur as a Poisson process
        r = np.random.RandomState(seed=self.randomness.get_seed())
        simulants_to_add = r.poisson(mean_births)

        if simulants_to_add > 0:
            self.simulant_creator(simulants_to_add,
                                  population_configuration={
                                      'age_start': 0,
                                      'age_end': 0,
                                      'sim_state': 'time_step',
                                  })

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
        self.randomness = builder.randomness.get_stream('fertility')
        asfr_data = builder.data.load("covariate.age_specific_fertility_rate.estimate")
        asfr_data = asfr_data[asfr_data.sex == 'Female'][['year_start', 'year_end',
                                                          'age_group_start', 'age_group_end', 'mean_value']]
        asfr_source = builder.lookup.build_table(asfr_data, key_columns=(),
                                                 parameter_columns=[('age', 'age_group_start', 'age_group_end'),
                                                                    ('year', 'year_start', 'year_end')],)
        self.asfr = builder.value.register_rate_producer('fertility rate', source=asfr_source)
        self.population_view = builder.population.get_view(['last_birth_time', 'sex', 'parent_id'])
        self.simulant_creator = builder.population.get_simulant_creator()
        builder.population.initializes_simulants(self.update_state_table,
                                                 creates_columns=['last_birth_time', 'parent_id'],
                                                 requires_columns=['sex'])

        builder.event.register_listener('time_step', self.step)

    def update_state_table(self, pop_data):
        """ Adds 'last_birth_time' and 'parent' columns to the state table."""

        women = self.population_view.get(pop_data.index, query="sex == 'Female'", omit_missing_columns=True).index
        last_birth_time = pd.Series(pd.NaT, name='last_birth_time', index=pop_data.index)

        # Do the naive thing, set so all women can have children
        # and none of them have had a child in the last year.
        last_birth_time[women] = pop_data.creation_time - pd.Timedelta(seconds=SECONDS_PER_YEAR)

        self.population_view.update(last_birth_time)
        self.population_view.update(pd.Series(-1, name='parent_id', index=pop_data.index, dtype=np.int64))

    def step(self, event):
        """Produces new children and updates parent status on time steps.
        Parameters
        ----------
        event : vivarium.population.PopulationEvent
            The event that triggered the function call.
        """
        # Get a view on all living women who haven't had a child in at least nine months.
        nine_months_ago = pd.Timestamp(event.time - PREGNANCY_DURATION)
        population = self.population_view.get(event.index, query='alive == "alive" and sex =="Female"')
        can_have_children = population.last_birth_time < nine_months_ago
        eligible_women = population[can_have_children]

        rate_series = self.asfr(eligible_women.index)
        had_children = self.randomness.filter_for_rate(eligible_women, rate_series).copy()

        had_children.loc[:, 'last_birth_time'] = event.time
        self.population_view.update(had_children['last_birth_time'])

        # If children were born, add them to the state table and record
        # who their mother was.
        num_babies = len(had_children)
        if num_babies:
            idx = self.simulant_creator(num_babies,
                                        population_configuration={
                                            'age_start': 0,
                                            'age_end': 0,
                                            'sim_state': 'time_step',
                                        })
            parents = pd.Series(data=had_children.index, index=idx, name='parent_id')
            self.population_view.update(parents)

    def __repr__(self):
        return "FertilityAgeSpecificRates()"
