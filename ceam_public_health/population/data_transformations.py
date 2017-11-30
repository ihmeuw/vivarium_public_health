from collections import namedtuple

import numpy as np


def assign_demographic_proportions(population_data):
    """Calculates conditional probabilities on the provided population data for use in sampling.

    Parameters
    ----------
    population_data : pandas.DataFrame
        Table with columns 'age', 'sex', 'year', 'location_id', and 'population'

    Returns
    -------
    pandas.DataFrame
        Table with the same columns as `population_data` and additionally with columns
        'P(sex, location_id, age| year)', 'P(sex, location_id | age, year)', and
        'P(age | year, sex, location_id)' with values calculated from the
        various population levels.
    """

    population_data['P(sex, location_id, age| year)'] = (
        population_data
            .groupby('year', as_index=False)
            .apply(lambda sub_pop: sub_pop.population / sub_pop[sub_pop.sex == 'Both'].population.sum())
            .reset_index(level=0).population
    )

    population_data['P(sex, location_id | age, year)'] = (
        population_data
            .groupby(['age', 'year'], as_index=False)
            .apply(lambda sub_pop: sub_pop.population / sub_pop[sub_pop.sex == 'Both'].population.sum())
            .reset_index(level=0).population
    )

    population_data['P(age | year, sex, location_id)'] = (
        population_data
            .groupby(['year', 'sex', 'location_id'], as_index=False)
            .apply(lambda sub_pop: sub_pop.population / sub_pop.population.sum())
            .reset_index(level=0).population
    )

    return population_data[population_data.sex != 'Both']


# TODO: Could probably clip the bins with the pdf calculated in smooth_ages rather than assuming
# a uniform distribution for this part.  The impact is probably minor though.
def rescale_binned_proportions(pop_data, pop_age_start, pop_age_end):
    """Clips the edge population data bins and rescales the proportions associated with those bins.

    Parameters
    ----------
    pop_data : pandas.DataFrame
        Table with columns 'age', 'age_group_start', 'age_group_end', 'sex', 'year',
        'location_id', 'population', 'P(sex, location_id, age| year)', 'P(sex, location_id | age, year)',
        'P(age | year, sex, location_id)'
    pop_age_start : float
        The starting age for the rescaled bins.
    pop_age_end : float
        The terminal age for the rescaled bins.

    Returns
    -------
    pandas.DataFrame
        Table with the same columns as `pop_data` where all bins outside the range
        (pop_age_start, pop_age_end) have been discarded.  If pop_age_start and pop_age_end
        don't fall cleanly on age boundaries, the bins in which they lie are clipped and
        the 'population', 'P(sex, location_id, age| year)', and 'P(age | year, sex, location_id)'
        values are rescaled to reflect their smaller representation.
    """
    age_start = max(pop_data.age_group_start.min(), pop_age_start)
    age_end = min(pop_data.age_group_end.max(), pop_age_end)

    relevant_age_groups = (pop_data.age_group_end > age_start) & (pop_data.age_group_start < age_end)
    pop_data = pop_data[relevant_age_groups].copy()

    for _, sub_pop in pop_data.groupby(['sex', 'location_id']):
        max_bin = sub_pop[sub_pop.age_group_end >= age_end]
        min_bin = sub_pop[sub_pop.age_group_start <= age_start]

        max_scale = ((age_end - float(max_bin.age_group_start))
                     / float(max_bin.age_group_end - max_bin.age_group_start))
        min_scale = ((float(min_bin.age_group_end) - age_start)
                     / float(min_bin.age_group_end - min_bin.age_group_start))

        columns_to_scale = ['P(sex, location_id, age| year)', 'P(age | year, sex, location_id)', 'population']

        pop_data.loc[max_bin.index, columns_to_scale] *= max_scale
        pop_data.loc[max_bin.index, 'age_group_end'] = age_end

        pop_data.loc[min_bin.index, columns_to_scale] *= min_scale
        pop_data.loc[min_bin.index, 'age_group_start'] = age_start

    return pop_data


AgeValues = namedtuple('AgeValues', ['current', 'young', 'old'])
EndpointValues = namedtuple('EndpointValues', ['left', 'right'])


def smooth_ages(simulants, population_data, randomness):
    """Distributes simulants among ages within their assigned age bins.

    Parameters
    ----------
    simulants : pandas.DataFrame
        Table with columns 'age', 'sex', and 'location'
    population_data : pandas.DataFrame
        Table with columns 'age', 'sex', 'year', 'location_id', 'population',
        'P(sex, location_id, age| year)', 'P(sex, location_id | age, year)',
        'P(age | year, sex, location_id)'
    randomness : vivarium.framework.randomness.RandomnessStream
        Source of random number generation within the vivarium common random number framework.

    Returns
    -------
    pandas.DataFrame
        Table with same columns as `simulants` with ages smoothed out within the age bins.
    """
    for (sex, location_id), sub_pop in population_data.groupby(['sex', 'location_id']):

        ages = sorted(sub_pop.age.unique())
        younger = [float(sub_pop.loc[sub_pop.age == ages[0], 'age_group_start'])] + ages[:-1]
        older = ages[1:] + [float(sub_pop.loc[sub_pop.age == ages[-1], 'age_group_end'])]

        uniform_all = randomness.get_draw(simulants.index, additional_key='smooth_ages')

        for age_set in zip(ages, younger, older):
            age = AgeValues(*age_set)

            has_correct_demography = ((simulants.age == age.current)
                                      & (simulants.sex == sex) & (simulants.location == location_id))
            affected = simulants[has_correct_demography]

            if affected.empty:
                continue

            # bin endpoints
            endpoints, proportions = _get_bins_and_proportions(sub_pop, age)
            pdf, slope, area, cdf_inflection_point = _construct_sampling_parameters(age, endpoints, proportions)

            # Make a draw from a uniform distribution
            uniform_rv = uniform_all.loc[affected.index]

            left_sims = affected[uniform_rv <= cdf_inflection_point]
            right_sims = affected[uniform_rv > cdf_inflection_point]

            simulants.loc[left_sims.index, 'age'] = _compute_ages(uniform_rv[left_sims.index],
                                                                  endpoints.left, pdf.left, slope.left, area)
            simulants.loc[right_sims.index, 'age'] = _compute_ages(uniform_rv[right_sims.index] - cdf_inflection_point,
                                                                   age.current, proportions.current, slope.right, area)

    return simulants


def _get_bins_and_proportions(pop_data, age):
    """Finds and returns the bin edges and the population proportions in the current and neighboring bins.

    Parameters
    ----------
    pop_data : pandas.DataFrame
        Table with columns 'age', 'sex', 'year', 'location_id', 'population',
        'P(sex, location_id, age| year)', 'P(sex, location_id | age, year)',
        'P(age | year, sex, location_id)'
    age : AgeValues
        Tuple with values
            (midpoint of current age bin, midpoint of previous age bin, midpoint of next age bin)

    Returns
    -------
    (EndpointValues, AgeValues)
        The `EndpointValues` tuple has values
            (age at left edge of bin, age at right edge of bin)
        The `AgeValues` tuple has values
            (proportion of pop in current bin, proportion of pop in previous bin, proportion of pop in next bin)
    """
    left = float(pop_data.loc[pop_data.age == age.current, 'age_group_start'])
    right = float(pop_data.loc[pop_data.age == age.current, 'age_group_end'])

    # proportion in this bin and the neighboring bins
    proportion_column = 'P(age | year, sex, location_id)'
    p_age = float(pop_data.loc[pop_data.age == age.current, proportion_column])
    p_young = float(pop_data.loc[pop_data.age == age.young, proportion_column]) if age.young != left else p_age
    p_old = float(pop_data.loc[pop_data.age == age.old, proportion_column]) if age.old != right else 0

    return EndpointValues(left, right), AgeValues(p_age, p_young, p_old)


def _construct_sampling_parameters(age, endpoint, proportion):
    """Calculates some sampling distribution parameters from known values.

    Parameters
    ----------
    age : AgeValues
        Tuple with values
            (midpoint of current age bin, midpoint of previous age bin, midpoint of next age bin)
    endpoint : EndpointValues
        Tuple with values
            (age at left edge of bin, age at right edge of bin)
    proportion : AgeValues
        Tuple with values
            (proportion of pop in current bin, proportion of pop in previous bin, proportion of pop in next bin)

    Returns
    -------
    (pdf, slope, area, cdf_inflection_point) : (EndpointValues, EndpointValues, float, float)
        pdf is a tuple with values
            (pdf evaluated at left bin edge, pdf evaluated at right bin edge)
        slope is a tuple with values
            (slope of pdf in left half bin, slope of pdf in right half bin)
        area is the total area under the pdf, used for normalization
        cdf_inflection_point is the value of the cdf at the midpoint of the age bin.
    """
    # pdf value at bin endpoints
    pdf_left = ((proportion.current - proportion.young) / (age.current - age.young)
                * (endpoint.left - age.young) + proportion.young)
    pdf_right = ((proportion.old - proportion.current) / (age.old - age.current)
                 * (endpoint.right - age.current) + proportion.current)
    pdf = EndpointValues(pdf_left, pdf_right)

    # normalization constant.  Total area under pdf.
    area = 0.5 * ((proportion.current + pdf.left) * (age.current - endpoint.left)
                  + (pdf.right + proportion.current) * (endpoint.right - age.current))

    # pdf slopes.
    m_left = (proportion.current - pdf.left) / (age.current - endpoint.left)
    m_right = (pdf.right - proportion.current) / (endpoint.right - age.current)
    slope = EndpointValues(m_left, m_right)

    # The decision bound on the uniform rv.
    cdf_inflection_point = 1 / (2 * area) * (proportion.current + pdf.left) * (age.current - endpoint.left)

    return pdf, slope, area, cdf_inflection_point


def _compute_ages(uniform_rv, start, height, slope, normalization):
    """Produces samples from the local age distribution.

    Parameters
    ----------
    uniform_rv : numpy.ndarray or float
        Values pulled from a uniform distribution and belonging to either the left or right half
        of the local distribution.  The halves are determined by the the point Z in [0, 1] such that
        Q(Z) = the midpoint of the age bin in question, where Q is inverse of the local
        cumulative distribution function.
    start : float
        Either the left edge of the age bin (if we're in the left half of the distribution) or
        the midpoint of the age bin (if we're in the right half of the distribution).
    height : float
        The value of the local distribution at `start`
    slope : float
        The slope of the local distribution.
    normalization : float
        The total area under the distribution.

    Returns
    -------
    numpy.ndarray or float
        Smoothed ages from one half of the age bin distribution.
    """
    if slope == 0:
        return start + normalization / height * uniform_rv
    else:
        return start + height / slope * (np.sqrt(1 + 2 * normalization * slope / height ** 2 * uniform_rv) - 1)


def get_cause_deleted_mortality(all_cause_mortality, list_of_csmrs):
    index_cols = ['age', 'sex', 'year']
    all_cause_mortality = all_cause_mortality.set_index(index_cols).copy()
    for csmr in list_of_csmrs:
        if csmr is None:
            continue
        all_cause_mortality = all_cause_mortality.subtract(csmr.set_index(index_cols)).dropna()
    return all_cause_mortality.reset_index().rename(columns={'rate': 'death_due_to_other_causes'})
