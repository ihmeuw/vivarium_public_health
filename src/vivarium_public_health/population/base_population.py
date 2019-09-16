"""
=========================
The Core Population Model
=========================

This module contains tools for sampling and assigning core demographic
characteristics to simulants.

"""
import pandas as pd
import numpy as np

from vivarium_public_health import utilities
from vivarium_public_health.population.data_transformations import (assign_demographic_proportions,
                                                                    rescale_binned_proportions,
                                                                    smooth_ages, load_population_structure)


class BasePopulation:
    """Component for producing and aging simulants based on demographic data."""

    configuration_defaults = {
        'population': {
            'age_start': 0,
            'age_end': 125,
            'exit_age': None,
        }
    }

    def __init__(self):
        self._sub_components = [AgeOutSimulants()]

    @property
    def name(self):
        return "base_population"

    @property
    def sub_components(self):
        return self._sub_components

    def setup(self, builder):
        self.config = builder.configuration.population
        input_config = builder.configuration.input_data

        self.randomness = {'general_purpose': builder.randomness.get_stream('population_generation'),
                           'bin_selection': builder.randomness.get_stream('bin_selection', for_initialization=True),
                           'age_smoothing': builder.randomness.get_stream('age_smoothing', for_initialization=True),
                           'age_smoothing_age_bounds': builder.randomness.get_stream('age_smoothing_age_bounds',
                                                                                     for_initialization=True)}
        self.register_simulants = builder.randomness.register_simulants

        columns = ['age', 'sex', 'alive', 'location', 'entrance_time', 'exit_time']

        self.population_view = builder.population.get_view(columns)
        builder.population.initializes_simulants(self.generate_base_population,
                                                 creates_columns=columns)

        source_population_structure = load_population_structure(builder)
        source_population_structure['location'] = input_config.location

        self.population_data = _build_population_data_table(source_population_structure)

        builder.event.register_listener('time_step', self.on_time_step, priority=8)

    @staticmethod
    def select_sub_population_data(reference_population_data, year):
        reference_years = sorted(set(reference_population_data.year_start))
        ref_year_index = np.digitize(year, reference_years).item()-1
        return reference_population_data[reference_population_data.year_start == reference_years[ref_year_index]]

    # TODO: Move most of this docstring to an rst file.
    def generate_base_population(self, pop_data):
        """Creates a population with fundamental demographic and simulation properties.

        When the simulation framework creates new simulants (essentially producing a new
        set of simulant ids) and this component is being used, the newly created simulants
        arrive here first and are assigned the demographic qualities 'age', 'sex', and 'location'
        in a way that is consistent with the demographic distributions represented by the
        population-level data.  Additionally, the simulants are assigned the simulation properties
        'alive', 'entrance_time', and 'exit_time'.

        The 'alive' parameter is alive or dead.
        In general, most simulation components (except for those computing summary statistics)
        ignore simulants if they are not in the 'alive' category. The 'entrance_time' and
        'exit_time' categories simply mark when the simulant enters or leaves the simulation,
        respectively.  Here we are agnostic to the methods of entrance and exit (e.g birth,
        migration, death, etc.) as these characteristics can be inferred from this column and
        other information about the simulant and the simulation parameters.

        Parameters
        ----------
        pop_data
        """

        age_params = {'age_start': pop_data.user_data.get('age_start', self.config.age_start),
                      'age_end': pop_data.user_data.get('age_end', self.config.age_end)}

        sub_pop_data = self.select_sub_population_data(self.population_data, pop_data.creation_time.year)

        self.population_view.update(generate_population(simulant_ids=pop_data.index,
                                                        creation_time=pop_data.creation_time,
                                                        step_size=pop_data.creation_window,
                                                        age_params=age_params,
                                                        population_data=sub_pop_data,
                                                        randomness_streams=self.randomness,
                                                        register_simulants=self.register_simulants))

    def on_time_step(self, event):
        """Ages simulants each time step.

        Parameters
        ----------
        event : vivarium.framework.population.PopulationEvent
        """
        population = self.population_view.get(event.index, query="alive == 'alive'")
        population['age'] += utilities.to_years(event.step_size)
        self.population_view.update(population)

    def __repr__(self):
        # TODO: Make a __str__ with some info about relevant config settings?
        return "BasePopulation()"


class AgeOutSimulants:
    """Component for handling aged-out simulants"""

    @property
    def name(self):
        return "age_out_simulants"

    def setup(self, builder):
        if builder.configuration.population.exit_age is None:
            return
        self.config = builder.configuration.population
        self.population_view = builder.population.get_view(['age', 'exit_time', 'tracked'])
        builder.event.register_listener('time_step__cleanup', self.on_time_step_cleanup)

    def on_time_step_cleanup(self, event):
        population = self.population_view.get(event.index)
        max_age = float(self.config.exit_age)
        pop = population[(population['age'] >= max_age) & population['tracked']].copy()
        if len(pop) > 0:
            pop['tracked'] = pd.Series(False, index=pop.index)
            pop['exit_time'] = event.time
            self.population_view.update(pop)

    def __repr__(self):
        return "AgeOutSimulants()"


def generate_population(simulant_ids, creation_time, step_size, age_params,
                        population_data, randomness_streams, register_simulants):
    """Produces a randomly generated set of simulants sampled from the provided `population_data`.

    Parameters
    ----------
    simulant_ids : iterable of ints
        Values to serve as the index in the newly generated simulant DataFrame.
    creation_time : pandas.Timestamp
        The simulation time when the simulants are created.
    age_params : dict
        Dictionary with keys
            age_start : Start of an age range
            age_end : End of an age range

        The latter two keys can have values specified to generate simulants over an age range.
    population_data : pandas.DataFrame
        Table with columns 'age', 'age_start', 'age_end', 'sex', 'year',
        'location', 'population', 'P(sex, location, age| year)', 'P(sex, location | age, year)'
    randomness_streams : Dict[str, vivarium.framework.randomness.RandomnessStream]
        Source of random number generation within the vivarium common random number framework.
    step_size : float
        The size of the initial time step.
    register_simulants : Callable
        A function to register the new simulants with the CRN framework.

    Returns
    -------
    simulants : pandas.DataFrame
        Table with columns
            'entrance_time'
                The `pandas.Timestamp` describing when the simulant entered
                the simulation. Set to `creation_time` for all simulants.
            'exit_time'
                The `pandas.Timestamp` describing when the simulant exited
                the simulation. Set initially to `pandas.NaT`.
            'alive'
                One of 'alive' or 'dead' indicating how the simulation
                interacts with the simulant.
            'age'
                The age of the simulant at the current time step.
            'location'
                The location indicating where the simulant resides.
            'sex'
                Either 'Male' or 'Female'.  The sex of the simulant.

    """
    simulants = pd.DataFrame({'entrance_time': pd.Series(creation_time, index=simulant_ids),
                              'exit_time': pd.Series(pd.NaT, index=simulant_ids),
                              'alive': pd.Series('alive', index=simulant_ids)},
                             index=simulant_ids)
    age_start = float(age_params['age_start'])
    age_end = float(age_params['age_end'])
    if age_start == age_end:
        return _assign_demography_with_initial_age(simulants, population_data, age_start,
                                                   step_size, randomness_streams, register_simulants)
    else:  # age_params['age_start'] is not None and age_params['age_end'] is not None
        return _assign_demography_with_age_bounds(simulants, population_data, age_start,
                                                  age_end, randomness_streams, register_simulants)


def _assign_demography_with_initial_age(simulants, pop_data, initial_age,
                                        step_size, randomness_streams, register_simulants):
    """Assigns age, sex, and location information to the provided simulants given a fixed age.

    Parameters
    ----------
    simulants : pandas.DataFrame
        Table that represents the new cohort of agents being added to the simulation.
    pop_data : pandas.DataFrame
        Table with columns 'age', 'age_start', 'age_end', 'sex', 'year',
        'location', 'population', 'P(sex, location, age| year)', 'P(sex, location | age, year)'
    initial_age : float
        The age to assign the new simulants.
    randomness_streams : Dict[str, vivarium.framework.randomness.RandomnessStream]
        Source of random number generation within the vivarium common random number framework.
    step_size : pandas.Timedelta
        The size of the initial time step.
    register_simulants : Callable
        A function to register the new simulants with the CRN framework.

    Returns
    -------
    pandas.DataFrame
        Table with same columns as `simulants` and with the additional columns 'age', 'sex',  and 'location'.
    """
    pop_data = pop_data[(pop_data.age_start <= initial_age) & (pop_data.age_end >= initial_age)]

    if pop_data.empty:
        raise ValueError('The age {} is not represented by the population data structure'.format(initial_age))

    age_fuzz = randomness_streams['age_smoothing'].get_draw(simulants.index) * utilities.to_years(step_size)
    simulants['age'] = initial_age + age_fuzz
    register_simulants(simulants[['entrance_time', 'age']])

    # Assign a demographically accurate location and sex distribution.
    choices = pop_data.set_index(['sex', 'location'])['P(sex, location | age, year)'].reset_index()
    decisions = randomness_streams['general_purpose'].choice(simulants.index,
                                                             choices=choices.index,
                                                             p=choices['P(sex, location | age, year)'])

    simulants['sex'] = choices.loc[decisions, 'sex'].values
    simulants['location'] = choices.loc[decisions, 'location'].values

    return simulants


def _assign_demography_with_age_bounds(simulants, pop_data, age_start, age_end, randomness_streams, register_simulants):
    """Assigns age, sex, and location information to the provided simulants given a range of ages.

    Parameters
    ----------
    simulants : pandas.DataFrame
        Table that represents the new cohort of agents being added to the simulation.
    pop_data : pandas.DataFrame
        Table with columns 'age', 'age_start', 'age_end', 'sex', 'year',
        'location', 'population', 'P(sex, location, age| year)', 'P(sex, location | age, year)'
    age_start, age_end : float
        The start and end of the age range of interest, respectively.
    randomness_streams : Dict[str, vivarium.framework.randomness.RandomnessStream]
        Source of random number generation within the vivarium common random number framework.
    register_simulants : Callable
        A function to register the new simulants with the CRN framework.

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
    sub_pop_data = pop_data[(pop_data.age_start >= age_start) & (pop_data.age_end <= age_end)]
    choices = sub_pop_data.set_index(['age', 'sex', 'location'])['P(sex, location, age| year)'].reset_index()
    decisions = randomness_streams['bin_selection'].choice(simulants.index,
                                                           choices=choices.index,
                                                           p=choices['P(sex, location, age| year)'])
    simulants['age'] = choices.loc[decisions, 'age'].values
    simulants['sex'] = choices.loc[decisions, 'sex'].values
    simulants['location'] = choices.loc[decisions, 'location'].values
    simulants = smooth_ages(simulants, pop_data, randomness_streams['age_smoothing_age_bounds'])
    register_simulants(simulants[['entrance_time', 'age']])
    return simulants


def _build_population_data_table(data):
    """Constructs a population data table for use as a population distribution over demographic characteristics.

    Parameters
    ----------
    data : pd.DataFrame
        Population structure data

    Returns
    -------
    pandas.DataFrame
        Table with columns
            'age' : Midpoint of the age group,
            'age_start' : Lower bound of the age group,
            'age_end' : Upper bound of the age group,
            'sex' : 'Male' or 'Female',
            'location' : location,
            'year' : Year,
            'population' : Total population estimate,
            'P(sex, location | age, year)' : Conditional probability of sex and location given age and year,
            'P(sex, location, age | year)' : Conditional probability of sex, location, and age given year,
            'P(age | year, sex, location)' : Conditional probability of age given year, sex, and location.
    """
    return assign_demographic_proportions(data)

