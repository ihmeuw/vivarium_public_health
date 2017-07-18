from itertools import product
from collections import namedtuple

import numpy as np


def assign_demographic_proportions(population_data):
    """Calculates conditional probabilities on the provided population data for use in sampling.

    Parameters
    ----------
    population_data : pandas.DataFrame
        Table with columns 'age', 'sex', 'year', 'location_id', and 'pop_scaled'

    Returns
    -------
    pandas.DataFrame
        Table with columns 'age', 'sex', 'year', 'location_id', 'pop_scaled',
        'P(sex, location_id, age| year)', 'P(sex, location_id | age, year)'
    """
    def normalize(sub_pop):
        return sub_pop.pop_scaled / sub_pop[sub_pop.sex == 'Both'].pop_scaled.sum()

    population_data['P(sex, location_id, age| year)'] = population_data.groupby(
        'year', as_index=False).apply(normalize).reset_index(level=0).pop_scaled
    population_data['P(sex, location_id | age, year)'] = population_data.groupby(
        ['age', 'year'], as_index=False).apply(normalize).reset_index(level=0).pop_scaled
    return population_data[population_data.sex != 'Both']


def rescale_binned_proportions(pop_data, pop_age_start, pop_age_end):
    """Clips the edge population data bins and rescales the proportions associated with those bins.

    Parameters
    ----------
    pop_data : pandas.DataFrame
    pop_age_start : float
    pop_age_end : float

    Returns
    -------
    pandas.DataFrame
    """

    if pop_age_start != pop_data.age_group_start.min():
        pop_data = pop_data[pop_data.age_group_end > pop_age_start]
    if pop_age_end != pop_data.age_group_end.max():
        pop_data = pop_data[pop_data.age_group_start < pop_age_end]

    for sex, location_id in product(['Male', 'Female'], pop_data.location_id.unique()):
        in_location_and_sex_group = (pop_data.sex == sex) & (pop_data.location_id == location_id)
        max_bin = pop_data[(pop_data.age_group_end >= pop_age_end) & in_location_and_sex_group]
        min_bin = pop_data[(pop_data.age_group_start <= pop_age_start) & in_location_and_sex_group]

        max_scale = (float(max_bin.age_group_end)
                     - pop_age_end/float(max_bin.age_group_end - max_bin.age_group_start))
        min_scale = (pop_age_start
                     - float(min_bin.age_group_start)/float(min_bin.age_group_end - min_bin.age_group_start))

        pop_data[pop_data.sex == sex].loc[max_bin.index, 'P(sex, location_id, age| year)'] *= max_scale
        pop_data[pop_data.sex == sex].loc[min_bin.index, 'P(sex, location_id, age| year)'] *= min_scale

    return pop_data


AgeValues = namedtuple('AgeValues', ['current', 'young', 'old'])
EndpointValues = namedtuple('EndpointValues', ['left', 'right'])


def smooth_ages(simulants, population_data, randomness):
    """Distributes simulants among ages within their assigned age bins.

    Parameters
    ----------
    simulants : pandas.DataFrame
    population_data : pandas.DataFrame
    randomness : vivarium.framework.randomness.RandomnessStream

    Returns
    -------
    pandas.DataFrame
    """
    for sex, location_id in product(['Male', 'Female'], population_data.location_id.unique()):
        pop_data = population_data[(population_data.sex == sex) & (population_data.location_id == location_id)]

        ages = sorted(pop_data.age.unique())
        younger = [0] + ages[:-1]
        older = ages[1:] + [float(pop_data.loc[pop_data.age == ages[-1], 'age_group_end'])]
        uniform_all = randomness.get_draw(simulants.index, additional_key='smooth_ages')

        for age_set in zip(ages, younger, older):
            age = AgeValues(*age_set)
            affected = simulants[(simulants.age == age.current)
                                 & (simulants.sex == sex)
                                 & (simulants.location == location_id)]
            # bin endpoints
            endpoints, proportions = _get_bins_and_proportions(pop_data, age)
            pdf, slope, area, cdf_inflection_point = _construct_sampling_parameters(age, endpoints, proportions)

            # Make a draw from a uniform distribution
            uniform_rv = uniform_all.iloc[affected.index]

            left_sims = affected[uniform_rv <= cdf_inflection_point]
            right_sims = affected[uniform_rv > cdf_inflection_point]

            simulants.loc[left_sims.index, 'age'] = _compute_ages(uniform_rv[left_sims.index],
                                                                  endpoints.left, pdf.left, slope.left, area)
            simulants.loc[right_sims.index, 'age'] = _compute_ages(uniform_rv[right_sims.index],
                                                                   endpoints.right, pdf.right, slope.right, area)

    return simulants


def _get_bins_and_proportions(pop_data, age):
    """
    Parameters
    ----------
    pop_data : pandas.DataFrame
    age : AgeValues

    Returns
    -------
    (EndpointValues, AgeValues)
    """
    left = float(pop_data.loc[pop_data.age == age.current, 'age_group_start'])
    right = float(pop_data.loc[pop_data.age == age.current, 'age_group_end'])

    # proportion in this bin and the neighboring bins
    proportion_column = 'P(sex, location_id, age| year)'
    p_age = float(pop_data.loc[pop_data.age == age.current, proportion_column])
    p_young = float(pop_data.loc[pop_data.age == age.young, proportion_column]) if age.young != left else p_age
    p_old = float(pop_data.loc[pop_data.age == age.old, proportion_column]) if age.old != right else 0

    return EndpointValues(left, right), AgeValues(p_age, p_young, p_old)


def _construct_sampling_parameters(age, endpoint, proportion):
    """
    Parameters
    ----------
    age : AgeValues
    endpoint : EndpointValues
    proportion : AgeValues

    Returns
    -------
    (pdf, slope, area, cdf_inflection_point) : (EndpointValues, EndpointValues, float, float)
    """
    # pdf value at bin endpoints
    pdf_left = ((proportion.current - proportion.young)/(age.current - age.young)
                * (endpoint.left - age.young) + proportion.young)
    pdf_right = ((proportion.old - proportion.current) / (age.old - age.current)
                 * (endpoint.right - age.current) + proportion.current)
    pdf = EndpointValues(pdf_left, pdf_right)

    # normalization constant.  Total area under pdf.
    area = 0.5 * ((proportion.current + pdf.left)*(age.current - endpoint.left)
                  + (pdf.right + proportion.current)*(endpoint.right - age.current))

    # pdf slopes.
    m_left = (proportion.current - pdf.left) / (age.current - endpoint.left)
    m_right = (pdf.right - proportion.current) / (endpoint.right - age.current)
    slope = EndpointValues(m_left, m_right)

    # The decision bound on the uniform rv.
    cdf_inflection_point = 1 / (2 * area) * (proportion.age + pdf.left) * (age.current - endpoint.left)

    return pdf, slope, area, cdf_inflection_point


def _compute_ages(uniform_rv, start, height, slope, normalization):
    """
    Parameters
    ----------
    uniform_rv : numpy.ndarray or float
    start : float
    height : float
    slope : float
    normalization : float

    Returns
    -------
    numpy.ndarray or float
    """
    if slope == 0:
        return start + normalization/height*uniform_rv
    else:
        return start + height/slope*(np.sqrt(1 + 2*normalization*slope / height**2 * uniform_rv) - 1)







