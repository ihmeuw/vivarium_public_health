"""This module contains several components that  model birth rates."""
import pandas as pd
import numpy as np
from vivarium_public_health.population.base_population import load_population_structure

SECONDS_PER_YEAR = 365.25*24*60*60
# TODO: Incorporate better data into gestational model (probably as a separate component)
PREGNANCY_DURATION = pd.Timedelta(days=9*30.5)


class FertilityDeterministic:
    """Deterministic model of births.

    Attributes
    ----------
    fractional_new_births : float
        A rolling record of the fractional part of new births generated
        each time-step that allows us to
    """

    configuration_defaults = {
        'fertility_deterministic': {
            'number_of_new_simulants_each_year': 1000,
        },
    }

    def __init__(self):
        self.fractional_new_births = 0

    def setup(self, builder):
        self.config = builder.configuration.fertility_deterministic
        self.simulant_creator = builder.population.get_simulant_creator()
        builder.event.register_listener('time_step', self.add_new_birth_cohort)

    def add_new_birth_cohort(self, event):
        """Deterministically adds a new set of simulants at every timestep
        based on a parameter in the configuration.

        Parameters
        ----------
        event : vivarium.population.PopulationEvent
            The event that triggered the function call.
        creator : method
            A function or method for creating a population.
        """

        # Assume births are uniformly distributed throughout the year.
        step_size = event.step_size/pd.Timedelta(seconds=1)
        simulants_to_add = (self.config.number_of_new_simulants_each_year*step_size/SECONDS_PER_YEAR
                            + self.fractional_new_births)
        self.fractional_new_births = simulants_to_add % 1
        simulants_to_add = int(simulants_to_add)
        if simulants_to_add > 0:
            self.simulant_creator(simulants_to_add,
                                  population_configuration={
                                      'age_start': 0,
                                      'age_end': 0,
                                  })


class FertilityCrudeBirthRate:
    """Population-level model of births using Crude Birth Rate.

    Attributes
    ----------
    randomness : `randomness.RandomStream`
        A named stream of random numbers bound to vivarium's common
        random number framework.

    Notes
    -----
    The OECD definition of Crude Birthrate can be found on their
    website_, while a more thorough discussion of fertility and
    birth rate models can be found on Wikipedia_ or in demography
    textbooks.

    .. _website: https://stats.oecd.org/glossary/detail.asp?ID=490
    .. _Wikipedia: https://en.wikipedia.org/wiki/Birth_rate
    """
    def setup(self, builder):

        self.birth_rate = self.get_crude_birth_rate(builder)

        self.randomness = builder.randomness.get_stream('crude_birth_rate')
        self.simulant_creator = builder.population.get_simulant_creator()
        builder.event.register_listener('time_step', self.add_new_birth_cohort)

    def add_new_birth_cohort(self, event):
        """Adds new simulants every time step based on the Crude Birth Rate
        and an assumption that birth is a Poisson process

        Parameters
        ----------
        event : vivarium.population.PopulationEvent
            The event that triggered the function call.
        creator : method
            A function or method for creating a population.

        Notes
        -----
        The method for computing the Crude Birth Rate employed here is
        approximate.

        """

        birth_rate = self.birth_rate.at[event.time.year]
        population_size = len(event.index)
        step_size = event.step_size / pd.Timedelta(seconds=1)

        mean_births = birth_rate*population_size*step_size/SECONDS_PER_YEAR
        # Assume births occur as a Poisson process
        r = np.random.RandomState(seed=self.randomness.get_seed())
        simulants_to_add = r.poisson(mean_births)
        if simulants_to_add > 0:
            self.simulant_creator(simulants_to_add,
                                  population_configuration={
                                      'age_start': 0,
                                      'age_end': 0,
                                  })

    @staticmethod
    def get_crude_birth_rate(builder):
        population_data = load_population_structure(builder)
        exit_age = builder.configuration.population.to_dict().get('exit_age', None)
        if exit_age:
            if builder.configuration.population.age_end != exit_age:
                raise ValueError('If you specify an exit age, the initial population age end must be the same '
                                 'for the crude birth rate calculation to work.')
            cut_bin = population_data[(population_data.age_group_start < exit_age)
                                      & (population_data.age_group_end >= exit_age)]
            cut_bin.value *= (exit_age - cut_bin.age_group_start) / (cut_bin.age_group_end - cut_bin.age_group_start)
            cut_bin.loc[:, 'age_group_end'] = exit_age
            population_data = population_data[population_data.age_group_end <= exit_age]
            population_data = pd.concat([population_data, cut_bin], ignore_index=True)

        population_data = population_data.groupby(['year_start'])['value'].sum()
        birth_data = builder.data.load("covariate.live_births_by_sex.estimate",
                                       future=builder.configuration.input_data.forecast)
        birth_data = (birth_data[birth_data.parameter == 'mean_value']
                      .drop('parameter', 'columns')
                      .groupby(['year_start'])['value'].sum())
        birth_rate = ((birth_data / population_data)
                      .reset_index()
                      .rename(columns={'year_start': 'year'})
                      .set_index('year')
                      .value)

        exceeds_data = builder.configuration.time.end.year > birth_rate.index.max()
        if exceeds_data:
            if builder.configuration.interpolation.extrapolate:
                new_index = pd.RangeIndex(birth_rate.index.min(), builder.configuration.time.end.year + 1)
                birth_rate = birth_rate.reindex(new_index, fill_value=birth_rate.at[birth_rate.index.max()])
            else:
                raise ValueError('Trying to extrapolate beyond the end of available birth data.')
        return birth_rate

    def birth_rate_producer(self, year):
        """Computes a crude birth rate from demographic data in a given year.

        Parameters
        ----------
        year : int
            The year we want the birth rate for.

        Returns
        -------
        float
            The crude birth rate of the population in the given year in
            births per person per year.
        """

        most_recent_data_year = min(max(self._population_data.year_start), max(self._birth_data.year_start))
        if year > most_recent_data_year:
            if not self.extrapolate:
                raise ValueError('You need to set extrapolate=True to run simulation for the future years')

            # FIXME: Here we fix the future birthrate to be same as the most available data. Fix it when we have
            # a better idea
            year = most_recent_data_year

        population_table = self._population_data.query("year_start == @year and sex == 'Both'")
        births = float(self._birth_data.query('sex == "Both"').set_index(['year_start']).loc[year].mean_value)

        if self.exit_age is not None:
            population = population_table.query("age < @self.exit_age").population.sum()
        else:
            population = population_table.population.sum()

        return births / population


class FertilityAgeSpecificRates:
    """
    A simulant-specific model for fertility and pregnancies.
    """

    def setup(self, builder):
        """ Setup the common randomness stream and
        age-specific fertility lookup tables.

        Parameters
        ----------
        builder : vivarium.engine.Builder
            Framework coordination object.

        """

        self.randomness = builder.randomness.get_stream('fertility')
        asfr_data = builder.data.load("covariate.age_specific_fertility_rate.estimate",
                                      future=builder.configuration.input_data.forecast)
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
        creator : method
            A function or method for creating a population.
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
                                        })
            parents = pd.Series(data=had_children.index, index=idx, name='parent_id')
            self.population_view.update(parents)
