from string import Template

import pandas as pd


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
    template = '{measure}'
    if by_year:
        template += '_in_{year}'
    if by_sex:
        template += '_among_{sex}'
    if by_age:
        template += '_in_age_group_{age_group}'
    return Template(template)


def get_group_counts(pop: pd.DataFrame, base_filter: str, base_key: Template,
                     config: dict, age_bins: pd.DataFrame = None) -> dict:
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
        ages = age_bins.iterrows()
        base_filter += ' and ({age_group_start} <= age) and (age < {age_group_end})'
    else:
        ages = [('all_ages', pd.Series({'age_group_start': None, 'age_group_end': None}))]

    if config['by_sex']:
        sexes = ['Male', 'Female']
        base_filter += ' and sex == {sex}'
    else:
        sexes = ['Both']

    group_counts = {}

    for group, age_group in ages:
        start, end = age_group.age_group_start, age_group.age_group_end
        for sex in sexes:
            filter_kwargs = {'age_group_start': start, 'age_group_end': end, 'sex': sex}
            key = base_key.safe_substitute(**filter_kwargs)
            group_filter = base_filter.format(**filter_kwargs)

            in_group = pop.query(group_filter)

            group_counts[key] = len(in_group)

    return group_counts


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
