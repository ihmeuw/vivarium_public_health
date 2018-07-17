import pandas as pd

import operator


def make_age_bins_column(age_group_id_min, age_group_id_max, builder):
    """
    Returns a dataframe with age bin information, including a new column called
    age_bin

    Parameters
    ----------
    age_group_id_min: int
        Youngest age group group id to be used in creating the columns
        (inclusive)

    age_group_id_max: int
        Oldest age group id to be used in creating the columns (inclusive)

    Returns
    -------
    New column (age_bin) is of the format "{lower_age_bound}_to_{upper_age_bound}".
    The upper and lower bounds are rounded to two decimal points
    """
    age_bins = builder.data.load("population.age_bins")
    age_bins = age_bins[(age_bins.age_group_id >= age_group_id_min) &
                        (age_bins.age_group_id <= age_group_id_max)]

    age_bins['age_bin'] = age_bins['age_group_name'].str.replace(" ", "_")

    return age_bins


def make_cols_demographically_specific(col_prefix, age_group_id_min, age_group_id_max, builder):
    """
    Returns a list of demographically specific (specific to an age group and sex)
    column names

    Parameters
    ----------
    col_prefix: str
        The name prefix of the column (see examples below for more
        clarification)

    age_group_id_min: int
        Youngest age group group id to be used in creating the columns
        (inclusive)

    age_group_id_max: int
        Oldest age group id to be used in creating the columns (inclusive)

    Examples
    --------
    make_cols_demographically_specific('diarrhea_event_count', 2, 5) returns a
    list of strings such as 'diarrhea_event_count_1_to_5_in_year_2010_among_Females'
    for each age and sex combination
    """
    age_bins = make_age_bins_column(age_group_id_min, age_group_id_max, builder)

    cols = []

    for age_bin in pd.unique(age_bins.age_bin.values):
        for sex in ['Male', 'Female']:
            cols.append('{c}_{a}_among_{s}s'.format(c=col_prefix,
                                                    a=age_bin,
                                                    s=sex))

    return cols


def make_age_bin_age_group_max_dict(age_group_id_min, age_group_id_max, builder):
    """
    Returns a dictionary where age_bin is the key and age group max is the
    value

    Parameters
    ----------
    age_group_id_min: int
        Youngest age group group id to be used in creating the columns
        (inclusive)

    age_group_id_max: int
        Oldest age group id to be used in creating the columns (inclusive)
    """
    age_bins = make_age_bins_column(age_group_id_min, age_group_id_max, builder)

    dict_of_age_bin_and_max_values = dict(zip(age_bins.age_bin,
                                              age_bins.age_group_years_end))

    sorted_dict = sorted(dict_of_age_bin_and_max_values.items(),
                         key=operator.itemgetter(1))

    return sorted_dict
