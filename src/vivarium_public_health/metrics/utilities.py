"""
=================
Metrics Utilities
=================

This module contains shared utilities for querying, parsing, and transforming
simulation data to support particular observations during the simulation.

"""
from collections import ChainMap
from string import Template
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.time import Time

from vivarium_public_health.disease.transition import TransitionString
from vivarium_public_health.utilities import to_years

_MIN_AGE = 0.0
_MAX_AGE = 150.0
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

    def __add__(self, other: Union[str, "QueryString"]) -> "QueryString":
        if self:
            if other:
                return QueryString(str(self) + " and " + str(other))
            else:
                return self
        else:
            return QueryString(other)

    def __radd__(self, other: Union[str, "QueryString"]) -> "QueryString":
        return QueryString(other) + self


class SubstituteString(str):
    """Normal string plus a no-op substitute method.

    Meant to be used with the OutputTemplate.

    """

    def substitute(self, *_, **__) -> "SubstituteString":
        """No-op method for consistency with OutputTemplate."""
        return self


class OutputTemplate(Template):
    """Output string template that enforces standardized formatting."""

    @staticmethod
    def format_template_value(value):
        """Formatting helper method for substituting values into a template."""
        return str(value).replace(" ", "_").lower()

    @staticmethod
    def get_mapping(*args, **kws):
        """Gets a consistent mapping from args passed to substitute."""
        # This is copied directly from the first part of Template.substitute
        if not args:
            raise TypeError(
                "descriptor 'substitute' of 'Template' object " "needs an argument"
            )
        self, *args = args  # allow the "self" keyword be passed
        if len(args) > 1:
            raise TypeError("Too many positional arguments")
        if not args:
            mapping = dict(kws)
        elif kws:
            mapping = ChainMap(kws, args[0])
        else:
            mapping = args[0]
        return self, mapping

    def substitute(*args, **kws) -> Union[SubstituteString, "OutputTemplate"]:
        """Substitutes provided values into the template.

        Users are allowed to pass any dictionary like object whose keys match
        placeholders in the template. Alternatively, they can provide
        keyword arguments where the keywords are the placeholders. If both
        are provided, the keywords take precedence.

        Returns
        -------
        Union[SubstituteString, OutputTemplate]
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


def get_age_bins(builder: Builder) -> pd.DataFrame:
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
    age_bins = builder.data.load("population.age_bins")

    # Works based on the fact that currently only models with age_start = 0 can include fertility
    age_start = builder.configuration.population.age_start
    min_bin_start = age_bins.age_start[np.asscalar(np.digitize(age_start, age_bins.age_end))]
    age_bins = age_bins[age_bins.age_start >= min_bin_start]
    age_bins.loc[age_bins.age_start < age_start, "age_start"] = age_start

    exit_age = builder.configuration.population.exit_age
    if exit_age:
        age_bins = age_bins[age_bins.age_start < exit_age]
        age_bins.loc[age_bins.age_end > exit_age, "age_end"] = exit_age
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
    OutputTemplate
        A template string with measure and possibly additional criteria.

    """
    template = "${measure}"
    if by_year:
        template += "_in_${year}"
    if by_sex:
        template += "_among_${sex}"
    if by_age:
        template += "_in_age_group_${age_group}"
    return OutputTemplate(template)


def get_age_sex_filter_and_iterables(
    config: Dict[str, bool], age_bins: pd.DataFrame, in_span: bool = False
) -> Tuple[QueryString, Tuple[List[Tuple[str, pd.Series]], List[str]]]:
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
    QueryString
        A filter on age and sex for use with DataFrame.query
    Tuple[List[Tuple[str, pd.Series]], List[str]]
        Iterables for the age and sex groups partially defining the bins
        for the observers.

    """
    age_sex_filter = QueryString("")
    if config["by_age"]:
        ages = list(age_bins.set_index("age_group_name").iterrows())
        if in_span:
            age_sex_filter += (
                "{age_start} < age_at_span_end and age_at_span_start < {age_end}"
            )
        else:
            age_sex_filter += "{age_start} <= age and age < {age_end}"
    else:
        ages = [("all_ages", pd.Series({"age_start": _MIN_AGE, "age_end": _MAX_AGE}))]

    if config["by_sex"]:
        sexes = ["Male", "Female"]
        age_sex_filter += 'sex == "{sex}"'
    else:
        sexes = ["Both"]

    return age_sex_filter, (ages, sexes)


def get_group_counts(
    pop: pd.DataFrame,
    base_filter: str,
    base_key: OutputTemplate,
    config: Dict[str, bool],
    age_bins: pd.DataFrame,
    aggregate: Callable[[pd.DataFrame], Union[int, float]] = len,
) -> Dict[Union[SubstituteString, OutputTemplate], Union[int, float]]:
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
    aggregate
        Transformation used to produce the aggregate.

    Returns
    -------
    Dict[Union[SubstituteString, OutputTemplate], Union[int, float]]
        A dictionary of output_key, count pairs where the output key is a
        string or template representing the sub groups.
    """
    age_sex_filter, (ages, sexes) = get_age_sex_filter_and_iterables(config, age_bins)
    base_filter += age_sex_filter

    group_counts = {}

    for group, age_group in ages:
        start, end = age_group.age_start, age_group.age_end
        for sex in sexes:
            filter_kwargs = {
                "age_start": start,
                "age_end": end,
                "sex": sex,
                "age_group": group,
            }
            group_key = base_key.substitute(**filter_kwargs)
            group_filter = base_filter.format(**filter_kwargs)
            in_group = pop.query(group_filter) if group_filter and not pop.empty else pop

            group_counts[group_key] = aggregate(in_group)

    return group_counts


def get_population_counts(
    pop: pd.DataFrame, config: Dict[str, bool], event_time: Time, age_bins: pd.DataFrame
) -> Dict[Union[SubstituteString, OutputTemplate], Union[int, float]]:
    config = config.copy()
    config["by_year"] = True  # This is always an annual point estimate
    base_key = get_output_template(**config).substitute(
        measure=f"population_count", year=event_time.year
    )
    base_filter = QueryString(f'alive == "alive"')
    return get_group_counts(pop, base_filter, base_key, config, age_bins)
