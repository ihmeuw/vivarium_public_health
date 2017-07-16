import pandas as pd

from vivarium import config
from vivarium.framework.event import listens_for
from vivarium.framework.population import uses_columns
from ceam_inputs import get_populations, get_subregions
from .data_transformations import assign_demographic_proportions, rescale_binned_proportions, smooth_ages


class BasePopulation:
    """Component for producing and aging simulants."""

    def __init__(self):
        main_location = config.simulation_parameters.location_id
        use_subregions = ('use_subregions' in config.simulation_parameters
                          and config.simulation_parameters.use_subregions)
        self._population_data = _build_population_data_table(main_location, use_subregions)

    def setup(self, builder):
        """
        Parameters
        ----------
        builder : vivarium.framework.engine.Builder
        """
        self.randomness = builder.randomness('population_generation')

    @listens_for('initialize_simulants', priority=0)
    @uses_columns(['age', 'sex', 'alive', 'location', 'entrance_time', 'exit_time'])
    def generate_base_population(self, event):
        """Creates a population with fundamental demographic and simulation properties.

        When the simulation framework creates new simulants (essentially producing a new
        set of simulant ids) and this component is being used, the newly created simulants
        arrive here first and are assigned the demographic qualities 'age', 'sex', and 'location'
        in a way that is consistent with the demographic distributions represented by the
        population-level data.  Additionally, the simulants are assigned the simulation properties
        'alive', 'entrance_time', and 'exit_time'.  The 'alive' parameter is categorical with categories
        {'alive', 'dead', and 'untracked'}.  In general, most simulation components (except for those
        computing summary statistics) ignore simulants if they are not in the 'alive' category.
        The 'entrance_time' and 'exit_time' categories simply mark when the simulant enters or leaves
        the simulation, respectively.  Here we are agnostic to the methods of entrance and exit (e.g
        birth, migration, death, etc.) as these characteristics can be inferred from this column and
        other information about the simulant and the simulation parameters.

        Parameters
        ----------
        event : vivarium.framework.population.PopulationEvent
        """
        age_params = {'initial_age': event.user_data.get('initial_age', None),
                      'pop_age_start': config.simulation_parameters.pop_age_start,
                      'pop_age_end': config.simulation_parameters.pop_age_start}
        sub_pop_data = self._population_data[self._population_data.year == event.time.year]
        event.population_view.update(generate_ceam_population(simulant_ids=event.index,
                                                              creation_time=event.time,
                                                              age_params=age_params,
                                                              population_data=sub_pop_data,
                                                              randomness_stream=self.randomness))

    @listens_for('time_step')
    @uses_columns(['age'], "alive == 'alive'")
    def age_simulants(self, event):
        time_step = config.simulation_parameters.time_step
        event.population['age'] += time_step / 365.0
        event.population_view.update(event.population)


@listens_for('time_step', priority=1)  # Set slightly after mortality.
@uses_columns(['alive', 'age', 'exit_time'], "alive == 'alive'")
def age_out_simulants(event):
    if 'maximum_age' not in config.simulation_parameters:
        raise ValueError('Must specify a maximum age in the config in order to use this component.')
    max_age = float(config.simulation_parameters.maximum_age)
    pop = event.population[event.population['age'] >= max_age].copy()
    pop['alive'] = 'untracked'
    pop['age'] = max_age
    pop['exit_time'] = pd.Timestamp(event.time)
    event.population_view.update(pop)


def generate_ceam_population(simulant_ids, creation_time, age_params, population_data, randomness_stream):
    """
    Parameters
    ----------
    simulant_ids : iterable of ints
    creation_time : datetime.datetime
    age_params : dict
    population_data : pandas.DataFrame
    randomness_stream : vivarium.framework.randomness.RandomnessStream

    Returns
    -------
    simulants : pandas.DataFrame
    """
    # TODO: Figure out if we actually use simulant_id anywhere and remove that dependency. It's a copy of the index.
    simulants = pd.DataFrame({'simulant_id': simulant_ids,
                              'entrance_time': pd.Series(pd.Timestamp(creation_time), index=simulant_ids),
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
    pop_data = pop_data[(pop_data.age_group_start <= initial_age) & (pop_data.age_group_end >= initial_age)]

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
    pop_data = rescale_binned_proportions(pop_data, age_start, age_end)

    # Assign a demographically accurate age, location, and sex distribution.
    choices = pop_data.set_index(['age', 'sex', 'location_id']).annual_proportion.reset_index()
    decisions = randomness_stream.choice(simulants.index,
                                         choices=choices.index,
                                         p=choices.annual_proportion)
    simulants['age'] = choices.loc[decisions, 'age'].values
    simulants['sex'] = choices.loc[decisions, 'sex'].values
    simulants['location'] = choices.loc[decisions, 'location_id'].values
    simulants['age'] = smooth_ages(simulants, pop_data, randomness_stream)

    return simulants


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
            `age` : Midpoint of the age group,
            `age_group_start` : Lower bound of the age group,
            `age_group_end` : Upper bound of the age group,
            `sex` : 'Male' or 'Female',
            `location_id` : GBD location id,
            `year` : Year,
            `pop_scaled` : Total population estimate,
            `P(sex, location_id | age, year)` : Conditional probability of sex and location_id given age and year,
            `P(sex, location_id, age | year)` : Conditional probability of sex, location_id, and age given year.
    """
    return assign_demographic_proportions(_get_population_data(main_location, use_subregions))


def _get_population_data(main_location, use_subregions):
    locations = get_subregions(main_location) if use_subregions else [main_location]
    return pd.concat([get_populations(location_id=location) for location in locations], ignore_index=True)

