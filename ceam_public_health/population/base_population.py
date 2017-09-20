import pandas as pd

from vivarium.framework.event import listens_for
from vivarium.framework.population import uses_columns

from ceam_inputs import get_populations, get_subregions

from .data_transformations import assign_demographic_proportions, rescale_binned_proportions, smooth_ages

SECONDS_PER_YEAR = 365.25*24*60*60


class BasePopulation:
    """Component for producing and aging simulants based on demographic data.

    Attributes
    ----------
    randomness : vivarium.framework.randomness.RandomnessStream
    """

    configuration_defaults = {
        'population': {
            'use_subregions': False,
            'initial_age': None,
            'pop_age_start': 0,
            'pop_age_end': 125,
            'maximum_age': None,
            'population_size': 10000,
        }
    }

    def setup(self, builder):
        """
        Parameters
        ----------
        builder : vivarium.framework.engine.Builder
        """
        self.randomness = builder.randomness('population_generation')
        self.config = builder.configuration.population
        input_config = builder.configuration.input_data

        self._population_data = _build_population_data_table(input_config.location_id, input_config.use_subregions)


    # TODO: Move most of this docstring to an rst file.
    @listens_for('initialize_simulants', priority=0)
    @uses_columns(['age', 'sex', 'alive', 'location', 'entrance_time', 'exit_time'])
    def generate_base_population(self, event):
        """Creates a population with fundamental demographic and simulation properties.

        When the simulation framework creates new simulants (essentially producing a new
        set of simulant ids) and this component is being used, the newly created simulants
        arrive here first and are assigned the demographic qualities 'age', 'sex', and 'location'
        in a way that is consistent with the demographic distributions represented by the
        population-level data.  Additionally, the simulants are assigned the simulation properties
        'alive', 'entrance_time', and 'exit_time'.

        The 'alive' parameter is categorical with categories {'alive', 'dead', and 'untracked'}.
        In general, most simulation components (except for those computing summary statistics)
        ignore simulants if they are not in the 'alive' category. The 'entrance_time' and
        'exit_time' categories simply mark when the simulant enters or leaves the simulation,
        respectively.  Here we are agnostic to the methods of entrance and exit (e.g birth,
        migration, death, etc.) as these characteristics can be inferred from this column and
        other information about the simulant and the simulation parameters.

        Parameters
        ----------
        event : vivarium.framework.population.PopulationEvent
        """
        age_params = {'initial_age': event.user_data.get('initial_age', None),
                      'pop_age_start': self.config.pop_age_start,
                      'pop_age_end': self.config.pop_age_end}
        sub_pop_data = self._population_data[self._population_data.year == event.time.year]
        event.population_view.update(generate_ceam_population(simulant_ids=event.index,
                                                              creation_time=event.time,
                                                              age_params=age_params,
                                                              population_data=sub_pop_data,
                                                              randomness_stream=self.randomness))

    @listens_for('time_step', priority=8)
    @uses_columns(['alive', 'age', 'exit_time'], "alive == 'alive'")
    def on_time_step(self, event):
        """Ages simulants each time step.

        Parameters
        ----------
        event : vivarium.framework.population.PopulationEvent
        """
        step_size = event.step_size/pd.Timedelta(seconds=1)
        event.population['age'] += step_size / SECONDS_PER_YEAR
        event.population_view.update(event.population)

        if self.config.maximum_age is not None:
            max_age = float(self.config.maximum_age)
            pop = event.population[event.population['age'] >= max_age].copy()
            pop['alive'] = pd.Series('untracked', index=pop.index).astype(
                'category', categories=['alive', 'dead', 'untracked'], ordered=False)
            pop['exit_time'] = event.time
            event.population_view.update(pop)


def generate_ceam_population(simulant_ids, creation_time, age_params, population_data, randomness_stream):
    """Produces a randomly generated set of simulants sampled from the provided `population_data`.

    Parameters
    ----------
    simulant_ids : iterable of ints
        Values to serve as the index in the newly generated simulant DataFrame.
    creation_time : pandas.Timestamp
        The simulation time when the simulants are created.
    age_params : dict
        Dictionary with keys
            initial_age : Fixed age to generate all simulants with (useful for, e.g., fertility)
            pop_age_start : Start of an age range
            pop_age_end : End of an age range
        The latter two keys can have values specified to generate simulants over an age range.
    population_data : pandas.DataFrame
        Table with columns 'age', 'age_group_start', 'age_group_end', 'sex', 'year',
        'location_id', 'pop_scaled', 'P(sex, location_id, age| year)', 'P(sex, location_id | age, year)'
    randomness_stream : vivarium.framework.randomness.RandomnessStream
        Source of random number generation within the vivarium common random number framework.

    Returns
    -------
    simulants : pandas.DataFrame
        Table with columns
            'entrance_time' : The `pandas.Timestamp` describing when the simulant entered
                the simulation. Set to `creation_time` for all simulants.
            'exit_time' : The `pandas.Timestamp` describing when the simulant exited
                the simulation. Set initially to `pandas.NaT`.
            'alive' : One of 'alive', 'dead', or 'untracked' indicating how the simulation
                interacts with the simulant.
            'age' : The age of the simulant at the current time step.
            'location' : The GBD location_id indicating where the simulant resides.
            'sex' : Either 'Male' or 'Female'.  The sex of the simulant.
    """
    simulants = pd.DataFrame({'entrance_time': pd.Series(creation_time, index=simulant_ids),
                              'exit_time': pd.Series(pd.NaT, index=simulant_ids),
                              'alive': pd.Series('alive', index=simulant_ids).astype(
                                  'category', categories=['alive', 'dead', 'untracked'], ordered=False)},
                             index=simulant_ids)

    if age_params['initial_age'] is not None:
        return _assign_demography_with_initial_age(simulants, population_data, float(age_params['initial_age']),
                                                   randomness_stream)
    else:  # age_params['pop_age_start'] is not None and age_params['pop_age_end'] is not None
        return _assign_demography_with_age_bounds(simulants, population_data, float(age_params['pop_age_start']),
                                                  float(age_params['pop_age_end']), randomness_stream)


def _assign_demography_with_initial_age(simulants, pop_data, initial_age, randomness_stream):
    """Assigns age, sex, and location information to the provided simulants given a fixed age.

    Parameters
    ----------
    simulants : pandas.DataFrame
        Table that represents the new cohort of agents being added to the simulation.
    pop_data : pandas.DataFrame
        Table with columns 'age', 'age_group_start', 'age_group_end', 'sex', 'year',
        'location_id', 'pop_scaled', 'P(sex, location_id, age| year)', 'P(sex, location_id | age, year)'
    initial_age : float
        The age to assign the new simulants.
    randomness_stream : vivarium.framework.randomness.RandomnessStream
        Source of random number generation within the vivarium common random number framework.

    Returns
    -------
    pandas.DataFrame
        Table with same columns as `simulants` and with the additional columns 'age', 'sex',  and 'location'.
    """
    pop_data = pop_data[(pop_data.age_group_start <= initial_age) & (pop_data.age_group_end >= initial_age)]

    if pop_data.empty:
        raise ValueError('The age {} is not represented by the population data structure'.format(initial_age))

    # Assign a demographically accurate location and sex distribution.
    choices = pop_data.set_index(['sex', 'location_id'])['P(sex, location_id | age, year)'].reset_index()
    decisions = randomness_stream.choice(simulants.index,
                                         choices=choices.index,
                                         p=choices['P(sex, location_id | age, year)'])

    simulants['age'] = initial_age
    simulants['sex'] = choices.loc[decisions, 'sex'].values
    simulants['location'] = choices.loc[decisions, 'location_id'].values

    return simulants


def _assign_demography_with_age_bounds(simulants, pop_data, age_start, age_end, randomness_stream):
    """Assigns age, sex, and location information to the provided simulants given a range of ages.

    Parameters
    ----------
    simulants : pandas.DataFrame
        Table that represents the new cohort of agents being added to the simulation.
    pop_data : pandas.DataFrame
        Table with columns 'age', 'age_group_start', 'age_group_end', 'sex', 'year',
        'location_id', 'pop_scaled', 'P(sex, location_id, age| year)', 'P(sex, location_id | age, year)'
    age_start, age_end : float
        The start and end of the age range of interest, respectively.
    randomness_stream : vivarium.framework.randomness.RandomnessStream
        Source of random number generation within the vivarium common random number framework.

    Returns
    -------
    pandas.DataFrame
        Table with same columns as `simulants` and with the additional columns 'age', 'sex',  and 'location'.
    """
    pop_data = rescale_binned_proportions(pop_data, age_start, age_end)

    if pop_data.empty:
        raise ValueError(
            'The age range ({}, {}) is not represented by the population data structure'.format(age_start, age_end))

    # Assign a demographically accurate age, location, and sex distribution.
    choices = pop_data.set_index(['age', 'sex', 'location_id'])['P(sex, location_id, age| year)'].reset_index()
    decisions = randomness_stream.choice(simulants.index,
                                         choices=choices.index,
                                         p=choices['P(sex, location_id, age| year)'])
    simulants['age'] = choices.loc[decisions, 'age'].values
    simulants['sex'] = choices.loc[decisions, 'sex'].values
    simulants['location'] = choices.loc[decisions, 'location_id'].values
    return smooth_ages(simulants, pop_data, randomness_stream)


def _build_population_data_table(main_location, use_subregions):
    """Constructs a population data table for use as a population distribution over demographic characteristics.

    Parameters
    ----------
    main_location : int
        The GBD location_id associated with the region being modeled.
    use_subregions : bool
        Whether the GBD subregion demography should be used in place of the main_location demography.

    Returns
    -------
    pandas.DataFrame
        Table with columns
            'age' : Midpoint of the age group,
            'age_group_start' : Lower bound of the age group,
            'age_group_end' : Upper bound of the age group,
            'sex' : 'Male' or 'Female',
            'location_id' : GBD location id,
            'year' : Year,
            'pop_scaled' : Total population estimate,
            'P(sex, location_id | age, year)' : Conditional probability of sex and location_id given age and year,
            'P(sex, location_id, age | year)' : Conditional probability of sex, location_id, and age given year,
            'P(age | year, sex, location_id)' : Conditional probability of age given year, sex, and location_id.
    """
    return assign_demographic_proportions(_get_population_data(main_location, use_subregions))


def _get_population_data(main_location, use_subregions):
    """Grabs all relevant population data from the GBD and returns it as a pandas DataFrame.

    Parameters
    ----------
    main_location : int
        The GBD location_id associated with the region being modeled.
    use_subregions : bool
        Whether the GBD subregion demography should be used in place of the main_location demography.

    Returns
    -------
    pandas.DataFrame
        Table with columns
            'age' : Midpoint of the age group,
            'age_group_start' : Lower bound of the age group,
            'age_group_end' : Upper bound of the age group,
            'sex' : 'Male' or 'Female',
            'location_id' : GBD location id,
            'year' : Year,
            'pop_scaled' : Total population estimate
    """
    locations = [main_location]
    if use_subregions:
        sub_regions = get_subregions(main_location)
        locations = sub_regions if sub_regions else locations
    return pd.concat([get_populations(location_id=location) for location in locations], ignore_index=True)
