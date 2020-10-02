"""
=================
Metrics Utilities
=================

This module contains shared utilities for querying, parsing, and transforming
simulation data to support particular observations during the simulation.

"""
from collections import ChainMap
from string import Template
from typing import Union, List, Tuple, Dict, Callable

import numpy as np
import pandas as pd
from vivarium.framework.lookup import LookupTable
from vivarium.framework.values import Pipeline

from vivarium_public_health.utilities import to_years

_MIN_AGE = 0.
_MAX_AGE = 150.
_MIN_YEAR = 1900
_MAX_YEAR = 2100


class QueryString(str):
    """Convenience string that forms logical conjunctions using addition.

    This class is meant to be used to create a logical statement for
    use with pandas ``query`` functions. It hides away the management
    of conjunctions and the fence posting problems that management creates.

    Examples
    --------
    >>> from vivarium_public_health.metrics.utilities import QueryString
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


class SubstituteString(str):
    """Normal string plus a no-op substitute method.

    Meant to be used with the OutputTemplate.

    """
    def substitute(self, *_, **__):
        """No-op method for consistency with OutputTemplate."""
        return self


class OutputTemplate(Template):
    """Output string template that enforces standardized formatting."""
    @staticmethod
    def format_template_value(value):
        """Formatting helper method for substituting values into a template."""
        return str(value).replace(' ', '_').lower()

    @staticmethod
    def get_mapping(*args, **kws):
        """Gets a consistent mapping from args passed to substitute."""
        # This is copied directly from the first part of Template.substitute
        if not args:
            raise TypeError("descriptor 'substitute' of 'Template' object "
                            "needs an argument")
        self, *args = args  # allow the "self" keyword be passed
        if len(args) > 1:
            raise TypeError('Too many positional arguments')
        if not args:
            mapping = dict(kws)
        elif kws:
            mapping = ChainMap(kws, args[0])
        else:
            mapping = args[0]
        return self, mapping

    def substitute(*args, **kws):
        """Substitutes provided values into the template.

        Users are allowed to pass any dictionary like object whose keys match
        placeholders in the template. Alternatively, they can provide
        keyword arguments where the keywords are the placeholders. If both
        are provided, the keywords take precedence.

        Returns
        -------
            Another output template with the provided values substituted in
            to the template placeholders if any placeholders remain,
            otherwise a completed ``SubstituteString``.
        """
        self, mapping = OutputTemplate.get_mapping(*args, **kws)
        mapping = {key: self.format_template_value(value) for key, value in mapping.items()}
        try:
            return SubstituteString(super(OutputTemplate, self).substitute(mapping))
        except KeyError:
            return OutputTemplate(super(OutputTemplate, self).safe_substitute(mapping))

    def safe_substitute(*args, **kws):
        """Alias to OutputTemplate.substitute."""
        self, mapping = OutputTemplate.get_mapping(*args, **kws)
        return self.substitute(mapping)

    def __repr__(self):
        return super(OutputTemplate, self).safe_substitute()


def get_age_bins(builder) -> pd.DataFrame:
    """Retrieves age bins relevant to the current simulation.

    Parameters
    ----------
    builder
        The simulation builder.

    Returns
    -------
        DataFrame with columns ``age_group_name``, ``age_start``,
        and ``age_end``.

    """
    age_bins = builder.data.load('population.age_bins')

    # Works based on the fact that currently only models with age_start = 0 can include fertility
    age_start = builder.configuration.population.age_start
    min_bin_start = age_bins.age_start[np.asscalar(np.digitize(age_start, age_bins.age_end))]
    age_bins = age_bins[age_bins.age_start >= min_bin_start]
    age_bins.loc[age_bins.age_start < age_start, 'age_start'] = age_start

    exit_age = builder.configuration.population.exit_age
    if exit_age:
        age_bins = age_bins[age_bins.age_start < exit_age]
        age_bins.loc[age_bins.age_end > exit_age, 'age_end'] = exit_age
    return age_bins


def get_output_template(by_age: bool, by_sex: bool, by_year: bool, **_) -> OutputTemplate:
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
    return OutputTemplate(template)


def get_age_sex_filter_and_iterables(config: Dict[str, bool], age_bins: pd.DataFrame, in_span: bool = False) -> (
        QueryString, Tuple[List[Tuple[str, pd.Series]], List[str]]):
    """Constructs a filter and a set of iterables for age and sex.

    The constructed filter and iterables are based on configuration for the
    observer component.

    Parameters
    ----------
    config
        A mapping with 'by_age' and 'by_sex' keys and boolean values
        indicating whether the observer is binning data by the respective
        categories.
    age_bins
        A table containing bin names and bin edges.
    in_span
        Whether the filter for age corresponds to living through an age
        group in a time span or uses a point estimate of age at a particular
        point in time.

    Returns
    -------
    age_sex_filter
        A filter on age and sex for use with DataFrame.query
    (ages, sexes)
        Iterables for the age and sex groups partially defining the bins
        for the observers.

    """
    age_sex_filter = QueryString("")
    if config['by_age']:
        ages = list(age_bins.set_index('age_group_name').iterrows())
        if in_span:
            age_sex_filter += '{age_start} < age_at_span_end and age_at_span_start < {age_end}'
        else:
            age_sex_filter += '{age_start} <= age and age < {age_end}'
    else:
        ages = [('all_ages', pd.Series({'age_start': _MIN_AGE, 'age_end': _MAX_AGE}))]

    if config['by_sex']:
        sexes = ['Male', 'Female']
        age_sex_filter += 'sex == "{sex}"'
    else:
        sexes = ['Both']

    return age_sex_filter, (ages, sexes)


def get_time_iterable(config: Dict[str, bool], sim_start: pd.Timestamp,
                      sim_end: pd.Timestamp) -> List[Tuple[str, Tuple[pd.Timestamp, pd.Timestamp]]]:
    """Constructs an iterable for time bins.

    The constructed iterable are based on configuration for the observer
    component.

    Parameters
    ----------
    config
        A mapping with 'by_year' and a boolean value indicating whether
        the observer is binning data by year.
    sim_start
        The time the simulation starts.
    sim_end
        The time the simulation ends.

    Returns
    -------
    time_spans
        Iterable for the time groups partially defining the bins
        for the observers.

    """
    if config['by_year']:
        time_spans = [(year, (pd.Timestamp(f'1-1-{year}'), pd.Timestamp(f'1-1-{year + 1}')))
                      for year in range(sim_start.year, sim_end.year + 1)]
    else:
        time_spans = [('all_years', (pd.Timestamp(f'1-1-{_MIN_YEAR}'), pd.Timestamp(f'1-1-{_MAX_YEAR}')))]
    return time_spans


def get_group_counts(pop: pd.DataFrame, base_filter: str, base_key: OutputTemplate,
                     config: Dict[str, bool], age_bins: pd.DataFrame,
                     aggregate: Callable = len) -> Dict[Union[SubstituteString, OutputTemplate], Union[int, float]]:
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
        A base filter term (e.g.: alive, susceptible to a particular disease)
        formatted to work with the query method of the provided population
        dataframe.
    base_key
        A template string with replaceable fields corresponding to the
        requested filters.
    config
        A dict with ``by_age`` and ``by_sex`` keys and boolean values.
    age_bins
        A dataframe with ``age_start`` and ``age_end`` columns.

    Returns
    -------
        A dictionary of output_key, count pairs where the output key is a
        string or template representing the sub groups.
    """
    age_sex_filter, (ages, sexes) = get_age_sex_filter_and_iterables(config, age_bins)
    base_filter += age_sex_filter

    group_counts = {}

    for group, age_group in ages:
        start, end = age_group.age_start, age_group.age_end
        for sex in sexes:
            filter_kwargs = {'age_start': start, 'age_end': end, 'sex': sex, 'age_group': group}
            group_key = base_key.substitute(**filter_kwargs)
            group_filter = base_filter.format(**filter_kwargs)
            in_group = pop.query(group_filter) if group_filter and not pop.empty else pop

            group_counts[group_key] = aggregate(in_group)

    return group_counts


def get_susceptible_person_time(pop, config, disease, current_year, step_size, age_bins):
    base_key = get_output_template(**config).substitute(measure=f'{disease}_susceptible_person_time', year=current_year)
    base_filter = QueryString(f'alive == "alive" and {disease} == "susceptible_to_{disease}"')
    person_time = get_group_counts(pop, base_filter, base_key, config, age_bins,
                                   aggregate=lambda x: len(x) * to_years(step_size))
    return person_time


def get_disease_event_counts(pop, config, disease, event_time, age_bins):
    base_key = get_output_template(**config).substitute(measure=f'{disease}_counts', year=event_time.year)
    # Can't use query with time stamps, so filter
    pop = pop.loc[pop[f'{disease}_event_time'] == event_time]
    base_filter = QueryString('')
    return get_group_counts(pop, base_filter, base_key, config, age_bins)


def get_prevalent_cases(pop, config, disease, event_time, age_bins):
    config = config.copy()
    config['by_year'] = True  # This is always an annual point estimate
    base_key = get_output_template(**config).substitute(measure=f'{disease}_prevalent_cases', year=event_time.year)
    base_filter = QueryString(f'alive == "alive" and {disease} != "susceptible_to_{disease}"')
    return get_group_counts(pop, base_filter, base_key, config, age_bins)


def get_population_counts(pop, config, event_time, age_bins):
    config = config.copy()
    config['by_year'] = True  # This is always an annual point estimate
    base_key = get_output_template(**config).substitute(measure=f'population_count', year=event_time.year)
    base_filter = QueryString(f'alive == "alive"')
    return get_group_counts(pop, base_filter, base_key, config, age_bins)


def get_person_time(pop: pd.DataFrame, config: Dict[str, bool], sim_start: pd.Timestamp,
                    sim_end: pd.Timestamp, age_bins: pd.DataFrame) -> Dict[str, float]:
    base_key = get_output_template(**config).substitute(measure='person_time')
    base_filter = QueryString("")
    time_spans = get_time_iterable(config, sim_start, sim_end)

    person_time = {}
    for year, (t_start, t_end) in time_spans:
        year_key = base_key.substitute(year=year)
        lived_in_span = get_lived_in_span(pop, t_start, t_end)
        person_time_in_span = get_person_time_in_span(lived_in_span, base_filter, year_key, config, age_bins)
        person_time.update(person_time_in_span)
    return person_time


def get_lived_in_span(pop: pd.DataFrame, t_start: pd.Timestamp, t_end: pd.Timestamp) -> pd.DataFrame:
    """Gets a subset of the population that lived in the time span.

    Parameters
    ----------
    pop
        A table representing the population with columns for 'entrance_time',
        'exit_time' and 'age'.
    t_start
        The date and time at the start of the span.
    t_end
        The date and time at the end of the span.

    Returns
    -------
    lived_in_span
        A table representing the population who lived some amount of time
        within the time span with all columns provided in the original
        table and additional columns 'age_at_span_start' and 'age_at_span_end'
        indicating the age of the individual at the start and end of the time
        span, respectively. 'age_at_span_start' will never be lower than the
        age at the simulant's entrance time and 'age_at_span_end' will never
        be greater than the age at the simulant's exit time.

    """
    lived_in_span = pop.loc[(t_start < pop['exit_time']) & (pop['entrance_time'] < t_end)]

    span_entrance_time = lived_in_span.entrance_time.copy()
    span_entrance_time.loc[t_start > span_entrance_time] = t_start
    span_exit_time = lived_in_span.exit_time.copy()
    span_exit_time.loc[t_end < span_exit_time] = t_end

    lived_in_span.loc[:, 'age_at_span_end'] = lived_in_span.age - to_years(lived_in_span.exit_time
                                                                           - span_exit_time)
    lived_in_span.loc[:, 'age_at_span_start'] = lived_in_span.age - to_years(lived_in_span.exit_time
                                                                             - span_entrance_time)
    return lived_in_span


def get_person_time_in_span(lived_in_span: pd.DataFrame, base_filter: QueryString,
                            span_key: OutputTemplate, config: Dict[str, bool],
                            age_bins: pd.DataFrame) -> Dict[Union[SubstituteString, OutputTemplate], float]:
    """Counts the amount of person time lived in a particular time span.

    Parameters
    ----------
    lived_in_span
        A table representing the subset of the population who lived in a
        particular time span. Must have columns for 'age_at_span_start' and
        'age_at_span_end'.
    base_filter
        A base filter term (e.g.: alive, susceptible to a particular disease)
        formatted to work with the query method of the provided population
        dataframe.
    span_key
        A template string with replaceable fields corresponding to the
        requested filters.
    config
        A dict with ``by_age`` and ``by_sex`` keys and boolean values.
    age_bins
        A dataframe with ``age_start`` and ``age_end`` columns.

    Returns
    -------
        A dictionary of output_key, person_time pairs where the output key
        corresponds to a particular demographic group.
    """
    person_time = {}
    age_sex_filter, (ages, sexes) = get_age_sex_filter_and_iterables(config, age_bins, in_span=True)
    base_filter += age_sex_filter

    for group, age_bin in ages:
        a_start, a_end = age_bin.age_start, age_bin.age_end
        for sex in sexes:
            filter_kwargs = {'sex': sex, 'age_start': a_start,
                             'age_end': a_end, 'age_group': group}

            key = span_key.substitute(**filter_kwargs)
            group_filter = base_filter.format(**filter_kwargs)

            in_group = lived_in_span.query(group_filter) if group_filter else lived_in_span.copy()
            age_start = np.maximum(in_group.age_at_span_start, a_start)
            age_end = np.minimum(in_group.age_at_span_end, a_end)

            person_time[key] = (age_end - age_start).sum()

    return person_time


def get_deaths(pop: pd.DataFrame, config: Dict[str, bool], sim_start: pd.Timestamp,
               sim_end: pd.Timestamp, age_bins: pd.DataFrame, causes: List[str]) -> Dict[str, int]:
    """Counts the number of deaths by cause.

    Parameters
    ----------
    pop
        The population dataframe to be counted. It must contain sufficient
        columns for any necessary filtering (e.g. the ``age`` column if
        filtering by age).
    config
        A dict with ``by_age``, ``by_sex``, and ``by_year`` keys and
        boolean values.
    sim_start
        The simulation start time.
    sim_end
        The simulation end time.
    age_bins
        A dataframe with ``age_start`` and ``age_end`` columns.
    causes
        List of causes present in the simulation.

    Returns
    -------
    deaths
        A dictionary of output_key, death_count pairs where the output_key
        represents a particular demographic subgroup.

    """
    base_filter = QueryString('alive == "dead" and cause_of_death == "death_due_to_{cause}"')
    base_key = get_output_template(**config)
    pop = clean_cause_of_death(pop)

    time_spans = get_time_iterable(config, sim_start, sim_end)

    deaths = {}
    for year, (t_start, t_end) in time_spans:
        died_in_span = pop[(t_start <= pop.exit_time) & (pop.exit_time < t_end)]
        for cause in causes:
            cause_year_key = base_key.substitute(measure=f'death_due_to_{cause}', year=year)
            cause_filter = base_filter.format(cause=cause)
            group_deaths = get_group_counts(died_in_span, cause_filter, cause_year_key, config, age_bins)
            deaths.update(group_deaths)
    return deaths


def get_years_of_life_lost(pop: pd.DataFrame, config: Dict[str, bool], sim_start: pd.Timestamp, sim_end: pd.Timestamp,
                           age_bins: pd.DataFrame, life_expectancy: LookupTable, causes: List[str]) -> Dict[str, float]:
    """Counts the years of life lost by cause.

    Parameters
    ----------
    pop
        The population dataframe to be counted. It must contain sufficient
        columns for any necessary filtering (e.g. the ``age`` column if
        filtering by age).
    config
        A dict with ``by_age``, ``by_sex``, and ``by_year`` keys and
        boolean values.
    sim_start
        The simulation start time.
    sim_end
        The simulation end time.
    age_bins
        A dataframe with ``age_start`` and ``age_end`` columns.
    life_expectancy
        A lookup table that takes in a pandas index and returns the life
        expectancy of the each individual represented by the index.
    causes
        List of causes present in the simulation.

    Returns
    -------
    years_of_life_lost
        A dictionary of output_key, yll_count pairs where the output_key
        represents a particular demographic subgroup.

    """
    base_filter = QueryString('alive == "dead" and cause_of_death == "death_due_to_{cause}"')
    base_key = get_output_template(**config)
    pop = clean_cause_of_death(pop)

    time_spans = get_time_iterable(config, sim_start, sim_end)

    years_of_life_lost = {}
    for year, (t_start, t_end) in time_spans:
        died_in_span = pop[(t_start <= pop.exit_time) & (pop.exit_time < t_end)]
        for cause in causes:
            cause_year_key = base_key.substitute(measure=f'ylls_due_to_{cause}', year=year)
            cause_filter = base_filter.format(cause=cause)
            group_ylls = get_group_counts(died_in_span, cause_filter, cause_year_key, config, age_bins,
                                          aggregate=lambda subgroup: sum(life_expectancy(subgroup.index)))
            years_of_life_lost.update(group_ylls)
    return years_of_life_lost


def get_years_lived_with_disability(pop: pd.DataFrame, config: Dict[str, bool], current_year: int,
                                    step_size: pd.Timedelta, age_bins: pd.DataFrame,
                                    disability_weights: Dict[str, Pipeline], causes: List[str]) -> Dict[str, float]:
    """Counts the years lived with disability by cause in the time step.

    Parameters
    ----------
    pop
        The population dataframe to be counted. It must contain sufficient
        columns for any necessary filtering (e.g. the ``age`` column if
        filtering by age).
    config
        A dict with ``by_age``, ``by_sex``, and ``by_year`` keys and
        boolean values.
    current_year
        The current year in the simulation.
    step_size
        The size of the current time step.
    age_bins
        A dataframe with ``age_start`` and ``age_end`` columns.
    disability_weights
        A mapping between causes and their disability weight pipelines.
    causes
        List of causes present in the simulation.

    Returns
    -------
    years_lived_with_disability
        A dictionary of output_key, yld_count pairs where the output_key
        represents a particular demographic subgroup.

    """
    base_key = get_output_template(**config).substitute(year=current_year)
    base_filter = QueryString('alive == "alive"')

    years_lived_with_disability = {}
    for cause in causes:
        cause_key = base_key.substitute(measure=f'ylds_due_to_{cause}')

        def count_ylds(sub_group):
            """Counts ylds attributable to a cause in the time step."""
            return sum(disability_weights[cause](sub_group.index) * to_years(step_size))

        group_ylds = get_group_counts(pop, base_filter, cause_key, config, age_bins, aggregate=count_ylds)
        years_lived_with_disability.update(group_ylds)

    return years_lived_with_disability


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
