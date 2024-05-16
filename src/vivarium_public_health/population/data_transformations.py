"""
===============================
Population Data Transformations
===============================

This module contains tools for handling raw demographic data and transforming
it into different distributions for sampling.

"""

from collections import namedtuple
from typing import Tuple, Union

import numpy as np
import pandas as pd
from vivarium.framework.randomness import RandomnessStream

_SORT_ORDER = ["location", "year_start", "sex", "age_start"]


def assign_demographic_proportions(
    population_data: pd.DataFrame,
    include_sex: str,
) -> pd.DataFrame:
    """Calculates conditional probabilities on the provided population data for sampling.

    Parameters
    ----------
    population_data
        Table with columns 'age', 'sex', 'year', 'location', and 'value'
    include_sex
        'Female', 'Male', or 'Both'.  Sexes to include in the distribution.

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
            'P(sex, location | age, year)' : Conditional probability of sex and
            location given age and year,
            'P(sex, location, age | year)' : Conditional probability of sex, location,
            and age given year,
            'P(age | year, sex, location)' : Conditional probability of age given
            year, sex, and location.
    """
    population_data = population_data.copy()
    if include_sex != "Both":
        population_data.loc[population_data.sex != include_sex, "value"] = 0.0

    population_data["P(sex, location, age| year)"] = (
        population_data.groupby("year_start", as_index=False)
        .apply(lambda sub_pop: sub_pop[["value"]] / sub_pop["value"].sum())
        .reset_index(level=0)["value"]
        .fillna(0.0)
    )

    population_data["P(sex, location | age, year)"] = (
        population_data.groupby(["age", "year_start"], as_index=False)
        .apply(lambda sub_pop: sub_pop[["value"]] / sub_pop["value"].sum())
        .reset_index(level=0)["value"]
        .fillna(0.0)
    )

    population_data["P(age | year, sex, location)"] = (
        population_data.groupby(["year_start", "sex", "location"], as_index=False)
        .apply(lambda sub_pop: sub_pop[["value"]] / sub_pop["value"].sum())
        .reset_index(level=0)["value"]
        .fillna(0.0)
    )

    return population_data.sort_values(_SORT_ORDER).reset_index(drop=True)


#  FIXME: This step should definitely be happening after we get some approximation of the
#   underlying distribution.  It makes an assumption of uniformity in the age bin.
#   This only happens at the edges, and is unlikely to be used to clip a neonatal age bin
#   where the impact would be significant.
def rescale_binned_proportions(
    pop_data: pd.DataFrame,
    age_start: float,
    age_end: float,
) -> pd.DataFrame:
    """Reshapes the distribution so that bin edges fall on the age_start and age_end.

    Parameters
    ----------
    pop_data
        Table with columns 'age', 'age_start', 'age_end', 'sex', 'year',
        'location', 'population', 'P(sex, location, age| year)',
        'P(sex, location | age, year)', 'P(age | year, sex, location)'
    age_start
        The starting age for the rescaled bins.
    age_end
        The terminal age for the rescaled bins.

    Returns
    -------
    pandas.DataFrame
        Table with the same columns as `pop_data` where all bins outside the range
        (age_start, age_end) have been discarded.  If age_start and age_end
        don't fall cleanly on age boundaries, the bins in which they lie are clipped and
        the 'population', 'P(sex, location, age| year)', and 'P(age | year, sex, location)'
        values are rescaled to reflect their smaller representation.
    """
    col_order = pop_data.columns.copy()
    if age_start > pop_data["age_end"].max():
        raise ValueError(
            "Provided population data is insufficient to model the requested age range."
        )

    age_start = max(pop_data["age_start"].min(), age_start)
    age_end = min(pop_data["age_end"].max(), age_end) - 1e-8
    pop_data = _add_edge_age_groups(pop_data.copy())

    columns_to_scale = [
        "P(sex, location, age| year)",
        "P(age | year, sex, location)",
        "value",
    ]
    for _, sub_pop in pop_data.groupby(["sex", "location"]):
        min_bin = sub_pop[
            (sub_pop["age_start"] <= age_start) & (age_start < sub_pop["age_end"])
        ]
        padding_bin = sub_pop[sub_pop["age_end"] == float(min_bin["age_start"].iloc[0])]

        min_scale = (float(min_bin["age_end"].iloc[0]) - age_start) / float(
            min_bin["age_end"].iloc[0] - min_bin["age_start"].iloc[0]
        )

        remainder = pop_data.loc[min_bin.index, columns_to_scale].values * (1 - min_scale)
        pop_data.loc[min_bin.index, columns_to_scale] *= min_scale
        pop_data.loc[padding_bin.index, columns_to_scale] += remainder

        pop_data.loc[min_bin.index, "age_start"] = age_start
        pop_data.loc[min_bin.index, "age"] = float(
            (
                pop_data.loc[min_bin.index, "age_start"].iloc[0]
                + pop_data.loc[min_bin.index, "age_end"].iloc[0]
            )
            / 2
        )
        pop_data.loc[padding_bin.index, "age_end"] = age_start
        pop_data.loc[padding_bin.index, "age"] = float(
            (
                pop_data.loc[padding_bin.index, "age_start"].iloc[0]
                + pop_data.loc[padding_bin.index, "age_end"].iloc[0]
            )
            / 2
        )

        max_bin = sub_pop[(sub_pop["age_end"] > age_end) & (age_end >= sub_pop["age_start"])]
        padding_bin = sub_pop[sub_pop["age_start"] == float(max_bin["age_end"].iloc[0])]

        max_scale = (age_end - float(max_bin["age_start"].iloc[0])) / float(
            max_bin["age_end"].iloc[0] - max_bin["age_start"].iloc[0]
        )

        remainder = pop_data.loc[max_bin.index, columns_to_scale] * (1 - max_scale)
        pop_data.loc[max_bin.index, columns_to_scale] *= max_scale
        pop_data.loc[padding_bin.index, columns_to_scale] += remainder.values

        pop_data.loc[max_bin.index, "age_end"] = age_end
        pop_data.loc[padding_bin.index, "age_start"] = age_end

    return pop_data[col_order].sort_values(_SORT_ORDER).reset_index(drop=True)


def _add_edge_age_groups(pop_data: pd.DataFrame) -> pd.DataFrame:
    """Pads the population data with age groups that enforce constant
    left interpolation and interpolation to zero on the right.

    """
    index_cols = ["location", "year_start", "year_end", "sex"]
    age_cols = ["age", "age_start", "age_end"]
    other_cols = [c for c in pop_data.columns if c not in index_cols + age_cols]
    pop_data = pop_data.set_index(index_cols)

    # For the lower bin, we want constant interpolation off the left side
    min_valid_age = pop_data["age_start"].min()
    # This bin width needs to be the same as the lowest bin.
    min_pad_age = min_valid_age - (pop_data["age_end"].min() - min_valid_age)
    min_pad_age_midpoint = (min_valid_age + min_pad_age) * 0.5

    lower_bin = pd.DataFrame(
        {"age_start": min_pad_age, "age_end": min_valid_age, "age": min_pad_age_midpoint},
        index=pop_data.index.unique(),
    )
    lower_bin[other_cols] = pop_data.loc[pop_data["age_start"] == min_valid_age, other_cols]

    # For the upper bin, we want our interpolation to go to zero.
    max_valid_age = pop_data["age_end"].max()
    # This bin width is not arbitrary.  It effects the rate at which our interpolation
    # zeros out. Since for the 2016 round the maximum age is 125, we're assuming almost no
    # one lives past that age, so we make this bin 1 week.  A more robust technique for
    # this would be better.
    max_pad_age = max_valid_age + 7 / 365
    max_pad_age_midpoint = (max_valid_age + max_pad_age) * 0.5

    upper_bin = pd.DataFrame(
        {"age_start": max_valid_age, "age_end": max_pad_age, "age": max_pad_age_midpoint},
        index=pop_data.index.unique(),
    )
    # We're doing the multiplication to ensure we get the correct data shape and index.
    upper_bin[other_cols] = 0 * pop_data.loc[pop_data["age_end"] == max_valid_age, other_cols]

    pop_data = pd.concat([lower_bin, pop_data, upper_bin], sort=False).reset_index()

    pop_data = pop_data.rename(
        columns={
            "level_0": "location",
            "level_1": "year_start",
            "level_2": "year_end",
            "level_3": "sex",
        }
    )
    return (
        pop_data[index_cols + age_cols + other_cols]
        .sort_values(by=["location", "year_start", "year_end", "age"])
        .reset_index(drop=True)
    )


AgeValues = namedtuple("AgeValues", ["current", "young", "old"])
EndpointValues = namedtuple("EndpointValues", ["left", "right"])


def smooth_ages(
    simulants: pd.DataFrame,
    population_data: pd.DataFrame,
    randomness: RandomnessStream,
) -> pd.DataFrame:
    """Distributes simulants among ages within their assigned age bins.

    Parameters
    ----------
    simulants
        Table with columns 'age', 'sex', and 'location'
    population_data
        Table with columns 'age', 'sex', 'year', 'location', 'population',
        'P(sex, location, age| year)', 'P(sex, location | age, year)',
        'P(age | year, sex, location)'
    randomness
        Source of random number generation within the vivarium common random number framework.

    Returns
    -------
    pandas.DataFrame
        Table with same columns as `simulants` with ages smoothed out within the age bins.
    """
    simulants = simulants.copy()
    for (sex, location), sub_pop in population_data.groupby(["sex", "location"]):
        ages = sorted(sub_pop["age"].unique())
        younger = [float(sub_pop.loc[sub_pop["age"] == ages[0], "age_start"].iloc[0])] + ages[
            :-1
        ]
        older = ages[1:] + [float(sub_pop.loc[sub_pop["age"] == ages[-1], "age_end"].iloc[0])]

        uniform_all = randomness.get_draw(simulants.index)

        for age_set in zip(ages, younger, older):
            age = AgeValues(*age_set)

            has_correct_demography = (
                (simulants["age"] == age.current)
                & (simulants["sex"] == sex)
                & (simulants["location"] == location)
            )
            affected = simulants[has_correct_demography]

            if affected.empty:
                continue

            # bin endpoints
            endpoints, proportions = _get_bins_and_proportions(sub_pop, age)
            pdf, slope, area, cdf_inflection_point = _construct_sampling_parameters(
                age, endpoints, proportions
            )

            # Make a draw from a uniform distribution
            uniform_rv = uniform_all.loc[affected.index]

            left_sims = affected[uniform_rv <= cdf_inflection_point]
            right_sims = affected[uniform_rv > cdf_inflection_point]

            simulants.loc[left_sims.index, "age"] = _compute_ages(
                uniform_rv[left_sims.index], endpoints.left, pdf.left, slope.left, area
            )
            simulants.loc[right_sims.index, "age"] = _compute_ages(
                uniform_rv[right_sims.index] - cdf_inflection_point,
                age.current,
                proportions.current,
                slope.right,
                area,
            )

    return simulants


def _get_bins_and_proportions(
    pop_data: pd.DataFrame,
    age: AgeValues,
) -> Tuple[EndpointValues, AgeValues]:
    """Finds and returns the bin edges and the population proportions in
    the current and neighboring bins.

    Parameters
    ----------
    pop_data
        Table with columns 'age', 'sex', 'year', 'location', 'population',
        'P(sex, location, age| year)', 'P(sex, location | age, year)',
        'P(age | year, sex, location)'
    age
        Tuple with values (
            midpoint of current age bin,
            midpoint of previous age bin,
            midpoint of next age bin,
        )

    Returns
    -------
    Tuple[EndpointValues, AgeValues]
        The `EndpointValues` tuple has values (
            age at left edge of bin,
            age at right edge of bin,
        )
        The `AgeValues` tuple has values (
            proportion of pop in current bin,
            proportion of pop in previous bin,
            proportion of pop in next bin,
        )

    """
    left = float(pop_data.loc[pop_data["age"] == age.current, "age_start"].iloc[0])
    right = float(pop_data.loc[pop_data["age"] == age.current, "age_end"].iloc[0])

    if not pop_data.loc[pop_data["age"] == age.young, "age_start"].empty:
        lower_left = float(pop_data.loc[pop_data["age"] == age.young, "age_start"].iloc[0])
    else:
        lower_left = left
    if not pop_data.loc[pop_data["age"] == age.old, "age_end"].empty:
        upper_right = float(pop_data.loc[pop_data["age"] == age.old, "age_end"].iloc[0])
    else:
        upper_right = right

    # proportion in this bin and the neighboring bins
    proportion_column = "P(age | year, sex, location)"
    # Here we make the assumption that
    # P(left < age < right | year, sex, location)  = p * (right - left)
    # in order to back out a point estimate for the probability density at the center
    # of the interval. This not the best assumption, but it'll do.
    p_age = float(
        pop_data.loc[pop_data["age"] == age.current, proportion_column].iloc[0]
        / (right - left)
    )
    p_young = (
        float(
            pop_data.loc[pop_data["age"] == age.young, proportion_column].iloc[0]
            / (left - lower_left)
        )
        if age.young != left
        else p_age
    )
    p_old = (
        float(
            pop_data.loc[pop_data["age"] == age.old, proportion_column].iloc[0]
            / (upper_right - right)
        )
        if age.old != right
        else 0
    )

    return EndpointValues(left, right), AgeValues(p_age, p_young, p_old)


def _construct_sampling_parameters(
    age: AgeValues, endpoint: EndpointValues, proportion: AgeValues
) -> Tuple[EndpointValues, EndpointValues, float, float]:
    """Calculates some sampling distribution parameters from known values.

    Parameters
    ----------
    age
        Tuple with values (
            midpoint of current age bin,
            midpoint of previous age bin,
            midpoint of next age bin,
        )
    endpoint : EndpointValues
        Tuple with values (
            age at left edge of bin,
            age at right edge of bin,
        )
    proportion : AgeValues
        Tuple with values (
            proportion of pop in current bin,
            proportion of pop in previous bin,
            proportion of pop in next bin,
        )

    Returns
    -------
    Tuple[EndpointValues, EndpointValues, float, float]
        A tuple of (pdf, slope, area, cdf_inflection_point) where
            pdf is a tuple with values (
                pdf evaluated at left bin edge,
                pdf evaluated at right bin edge,
            )
            slope is a tuple with values (
                slope of pdf in left half bin,
                slope of pdf in right half bin,
            )
            area is the total area under the pdf, used for normalization
            cdf_inflection_point is the value of the cdf at the midpoint of the age bin.

    """
    # pdf value at bin endpoints
    pdf_left = (proportion.current - proportion.young) / (age.current - age.young) * (
        endpoint.left - age.young
    ) + proportion.young
    pdf_right = (proportion.old - proportion.current) / (age.old - age.current) * (
        endpoint.right - age.current
    ) + proportion.current
    area = 0.5 * (
        (proportion.current + pdf_left) * (age.current - endpoint.left)
        + (pdf_right + proportion.current) * (endpoint.right - age.current)
    )

    pdf = EndpointValues(pdf_left, pdf_right)

    # pdf slopes.
    m_left = (proportion.current - pdf.left) / (age.current - endpoint.left)
    m_right = (pdf.right - proportion.current) / (endpoint.right - age.current)
    slope = EndpointValues(m_left, m_right)

    # The decision bound on the uniform rv.
    cdf_inflection_point = (
        1 / (2 * area) * (proportion.current + pdf.left) * (age.current - endpoint.left)
    )

    return pdf, slope, area, cdf_inflection_point


def _compute_ages(
    uniform_rv: Union[np.ndarray, float],
    start: float,
    height: float,
    slope: float,
    normalization: float,
) -> Union[np.ndarray, float]:
    """Produces samples from the local age distribution.

    Parameters
    ----------
    uniform_rv
        Values pulled from a uniform distribution and belonging to either the left or
        right half of the local distribution. The halves are determined by the point Z
        in [0, 1] such that Q(Z) = the midpoint of the age bin in question, where Q is
        inverse of the local cumulative distribution function.
    start
        Either the left edge of the age bin (if we're in the left half of the distribution) or
        the midpoint of the age bin (if we're in the right half of the distribution).
    height
        The value of the local distribution at `start`
    slope
        The slope of the local distribution.
    normalization
        The total area under the distribution.

    Returns
    -------
    Union[np.ndarray, float]
        Smoothed ages from one half of the age bin distribution.
    """
    if abs(slope) < np.finfo(np.float32).eps:
        return start + normalization / height * uniform_rv
    else:
        return start + height / slope * (
            np.sqrt(1 + 2 * normalization * slope / height**2 * uniform_rv) - 1
        )


def get_cause_deleted_mortality_rate(all_cause_mortality_rate, list_of_csmrs):
    index_cols = ["age_start", "age_end", "sex", "year_start", "year_end"]
    all_cause_mortality_rate = all_cause_mortality_rate.set_index(index_cols).copy()
    for csmr in list_of_csmrs:
        if csmr is None:
            continue
        all_cause_mortality_rate = all_cause_mortality_rate.subtract(
            csmr.set_index(index_cols)
        ).dropna()

    return all_cause_mortality_rate.reset_index().rename(
        columns={"value": "death_due_to_other_causes"}
    )


def load_population_structure(builder):
    data = builder.data.load("population.structure")
    # create an age column which is the midpoint of the age group
    data["age"] = data.apply(lambda row: (row["age_start"] + row["age_end"]) / 2, axis=1)
    data["location"] = builder.data.load("population.location")
    return data


def get_live_births_per_year(builder):
    population_data = load_population_structure(builder)
    birth_data = builder.data.load("covariate.live_births_by_sex.estimate")

    validate_crude_birth_rate_data(builder, population_data.year_end.max())
    population_data = rescale_final_age_bin(builder, population_data)

    initial_population_size = builder.configuration.population.population_size
    population_data = population_data.groupby(["year_start"])["value"].sum()
    birth_data = (
        birth_data[birth_data.parameter == "mean_value"]
        .drop(columns=["parameter"])
        .groupby(["year_start"])["value"]
        .sum()
    )

    start_year = builder.configuration.time.start.year
    if (
        builder.configuration.interpolation.extrapolate
        and start_year > birth_data.index.max()
    ):
        start_year = birth_data.index.max()

    if not builder.configuration.fertility.time_dependent_live_births:
        birth_data = birth_data.at[start_year]

    if not builder.configuration.fertility.time_dependent_population_fraction:
        population_data = population_data.at[start_year]

    live_birth_rate = initial_population_size / population_data * birth_data

    if isinstance(live_birth_rate, (int, float)):
        live_birth_rate = pd.Series(
            live_birth_rate,
            index=pd.RangeIndex(
                builder.configuration.time.start.year,
                builder.configuration.time.end.year + 1,
                name="year",
            ),
        )
    else:
        live_birth_rate = (
            live_birth_rate.reset_index()
            .rename(columns={"year_start": "year"})
            .set_index("year")
            .value
        )
        exceeds_data = builder.configuration.time.end.year > live_birth_rate.index.max()
        if exceeds_data:
            new_index = pd.RangeIndex(
                live_birth_rate.index.min(), builder.configuration.time.end.year + 1
            )
            live_birth_rate = live_birth_rate.reindex(
                new_index, fill_value=live_birth_rate.at[live_birth_rate.index.max()]
            )
    return live_birth_rate


def rescale_final_age_bin(builder, population_data):
    exit_age = builder.configuration.population.to_dict().get("exit_age", None)
    if exit_age:
        population_data = population_data.loc[population_data["age_start"] < exit_age].copy()
        cut_bin_idx = exit_age <= population_data["age_end"]
        cut_age_start = population_data.loc[cut_bin_idx, "age_start"]
        cut_age_end = population_data.loc[cut_bin_idx, "age_end"]
        population_data.loc[cut_bin_idx, "value"] *= (exit_age - cut_age_start) / (
            cut_age_end - cut_age_start
        )
        population_data.loc[cut_bin_idx, "age_end"] = exit_age
    return population_data


def validate_crude_birth_rate_data(builder, data_year_max):
    exit_age = builder.configuration.population.to_dict().get("exit_age", None)
    if exit_age and builder.configuration.population.age_end != exit_age:
        raise ValueError(
            "If you specify an exit age, the initial population age end must be the same "
            "for the crude birth rate calculation to work."
        )

    exceeds_data = builder.configuration.time.end.year > data_year_max
    if exceeds_data and not builder.configuration.interpolation.extrapolate:
        raise ValueError("Trying to extrapolate beyond the end of available birth data.")
