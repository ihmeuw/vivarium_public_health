from string import Template
from typing import Union

import pandas as pd


class QueryString(str):
    """Convenience string that forms logical conjunctions using addition.

    This class is meant to be used to create a logical statement for
    use with pandas ``query`` functions. It hides away the management
    of conjunctions and the fence posting problems that management creates.

    Examples
    --------
    >>> s = QueryString('')
    >>> s
    ''
    >>> s + ''
    ''
    >>> s + 'abc'
    'abc'
    >>> s += 'abc'
    >>> s + 'def'
    'abc and def'

    """
    def __add__(self, other: Union[str, 'QueryString']) -> 'QueryString':
        if self:
            if other:
                return QueryString(str(self) + ' and ' + str(other))
            else:
                return self
        else:
            return QueryString(other)

    def __radd__(self, other: Union[str, 'QueryString']) -> 'QueryString':
        return QueryString(other) + self


def get_age_bins(builder) -> pd.DataFrame:
    """Retrieves age bins relevant to the current simulation.

    Parameters
    ----------
    builder
        The simulation builder.

    Returns
    -------
        DataFrame with columns ``age_group_name``, ``age_group_start``,
        and ``age_group_end``.

    """
    age_bins = builder.data.load('population.age_bins')
    exit_age = builder.configuration.population.exit_age
    if exit_age:
        age_bins = age_bins[age_bins.age_group_start < exit_age]
        age_bins.loc[age_bins.age_group_end > exit_age, 'age_group_end'] = exit_age
    return age_bins


def get_output_template(by_age: bool, by_sex: bool, by_year: bool) -> Template:
    """Gets a template string for output metrics.

    The template string should be filled in using filter criteria for
    measure, age, sex, and year in the observer using this function.

    Parameters
    ----------
    by_age
        Whether the template should include age criteria.
    by_sex
        Whether the template should include sex criteria.
    by_year
        Whether the template should include year criteria.

    Returns
    -------
        A template string with measure and possibly additional criteria.

    """
    template = '${measure}'
    if by_year:
        template += '_in_${year}'
    if by_sex:
        template += '_among_${sex}'
    if by_age:
        template += '_in_age_group_${age_group}'
    return Template(template)


def get_group_counts(pop: pd.DataFrame, base_filter: str, base_key: Template,
                     config: dict, age_bins: pd.DataFrame) -> dict:
    """Gets a count of people in a custom subgroup.

    The user is responsible for providing a default filter (e.g. only alive
    people, or people susceptible to a particular disease).  Demographic
    filters will be applied based on standardized configuration.

    Parameters
    ----------
    pop
        The population dataframe to be counted.  It must contain sufficient
        columns for any necessary filtering (e.g. the ``age`` column if
        filtering by age).
    base_filter
        A base filter term (alive, susceptible to a particular disease)
        formatted to work with the query method of the provided population
        dataframe.
    base_key
        A template string with replaceable fields corresponding to the
        requested filters.
    config
        A dict with ``by_age`` and ``by_sex`` keys and boolean values.
    age_bins
        A dataframe with ``age_group_start`` and ``age_group_end`` columns.
        Only required if sub-setting people by age.

    Returns
    -------
        A dictionary of output_key, count pairs where the output key is a
        string template with an unfilled measure parameter.
    """
    if config['by_age']:
        ages = age_bins.set_index('age_group_name').iterrows()
        base_filter += '{age_group_start} <= age and age < {age_group_end}'
    else:
        ages = [('all_ages', pd.Series({'age_group_start': None, 'age_group_end': None, 'age_group_name': 'all_ages'}))]

    if config['by_sex']:
        sexes = ['Male', 'Female']
        base_filter += 'sex == "{sex}"'
    else:
        sexes = ['Both']

    group_counts = {}

    for group, age_group in ages:
        start, end = age_group.age_group_start, age_group.age_group_end
        for sex in sexes:
            filter_kwargs = {'age_group_start': start, 'age_group_end': end,
                             'sex': sex}
            template_kwargs = {'sex': sex.lower(), 'age_group': group.lower()}
            key = Template(base_key.safe_substitute(**template_kwargs))
            group_filter = base_filter.format(**filter_kwargs)
            in_group = pop.query(group_filter) if group_filter else pop

            group_counts[key] = len(in_group)

    return group_counts


def get_susceptible_person_time(pop, config, disease, current_year, step_size, age_bins):
    base_key = Template(get_output_template(**config).safe_substitute(year=current_year))
    base_filter = QueryString(f'alive == "alive" and {disease} == "susceptible_to_{disease}"')
    group_counts = get_group_counts(pop, base_filter, base_key, config, age_bins)

    person_time = {}
    for key, count in group_counts.items():
        person_time_key = key.safe_substitute(measure=f'{disease}_susceptible_person_time')
        person_time[person_time_key] = count * to_years(step_size)

    return person_time


def get_disease_event_counts(pop, config, disease, event_time, age_bins):
    base_key = Template(get_output_template(**config).safe_substitute(year=event_time.year))
    # Can't use query with time stamps, so filter
    pop = pop.loc[pop[f'{disease}_event_time'] == event_time]
    base_filter = QueryString('')

    group_counts = get_group_counts(pop, base_filter, base_key, config, age_bins)

    disease_events = {}
    for key, count in group_counts.items():
        count_key = key.safe_substitute(measure=f'{disease}_counts')
        disease_events[count_key] = count

    return disease_events


def clean_cause_of_death(pop: pd.DataFrame) -> pd.DataFrame:
    """Standardizes cause of death names to all read ``death_due_to_cause``."""

    def _clean(cod: str) -> str:
        if 'death' in cod or 'dead' in cod:
            pass
        else:
            cod = f'death_due_to_{cod}'
        return cod

    pop.cause_of_death = pop.cause_of_death.apply(_clean)
    return pop


def to_years(time: pd.Timedelta) -> float:
    """Converts a time delta to a float for years."""
    return time / pd.Timedelta(days=365.25)
